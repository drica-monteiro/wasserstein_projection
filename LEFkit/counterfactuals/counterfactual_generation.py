import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk

from sklearn import preprocessing

import sys

from tqdm import tqdm

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# AUTO-ENCODER FOR TABULAR DATA
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class autoencoder_explicit_S_tabularData(nn.Module):
    """
    Autoencoder architecture for tabular data in which the parameters of the decoder depend on S.
     -> Denoted 'S-informed AE'
     -> the latent space definition is the same for all data but its interpetation depends on the
       group of the data
    """
    def __init__(self,nb_input_var,latent_space_dim):
        super(autoencoder_explicit_S_tabularData, self).__init__()
        

        self.encoder = nn.Sequential(
            nn.Linear(nb_input_var,256),
            nn.ReLU(True),
            nn.Linear(256,256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.ReLU(True),
            nn.Linear(256,256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.ReLU(True),
            nn.Linear(256,latent_space_dim)
        )

        self.decoder_S0 = nn.Sequential(
            nn.Linear(latent_space_dim,256),
            nn.ReLU(True),
            nn.Linear(256,256),
            nn.ReLU(True),
            nn.Linear(256,nb_input_var)
        )

        self.decoder_S1 = nn.Sequential(
            nn.Linear(latent_space_dim,256),
            nn.ReLU(True),
            nn.Linear(256,256),
            nn.ReLU(True),
            nn.Linear(256,nb_input_var)
        )
        
        self.S_predictor = nn.Sequential(
            nn.Linear(latent_space_dim,1),
            nn.Sigmoid()
        )



    def forward(self, x):
        #1) encoder + mask definition
        latent = self.encoder(x)

        #2) decoders
        x0 = self.decoder_S0(latent)
        x1 = self.decoder_S1(latent)
        predS = self.S_predictor(latent)

        return x0,x1,predS

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# FUNCTION TO TRAIN autoencoder_explicit_S_tabularData
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def fit_AE_SensitiveImpact(model,X_train_in,X_train_out,labels_train,X_test,labels_test,DEVICE,num_epochs,batch_size,nb_add_obs_in_S0=3,learning_rate = 1e-3):
  """
  Fit function for the S-informed AE autoencoder_explicit_S_adult.
  -> Globally follows the epochs/mini-batch training standards
  -> Can add randomly drawn observations in S0 in each mini-batch
  """
  
  
  criterion_S0 = nn.MSELoss()
  criterion_S1 = nn.MSELoss()
  #criterion = nn.BCELoss()  #if the outputs are probabilities
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

  dim_X=len(X_train_in.shape)  #dim 0 refers to the mini-batch observations / the other dimensions refer to each observation dimension... for instance dim_X=4 for a mini-bach of multi-channel 2D images [mb,channels,dimX,dimY]
  n_train=X_train_in.shape[0]
  n_test=X_test.shape[0]

  conv_losses_train=[]
  conv_losses_test=[]




  for epoch in tqdm(range(num_epochs)):
    #print('epoch:',epoch)
    obsIDs=np.arange(n_train)
    np.random.shuffle(obsIDs)


    for i in range(0,n_train-batch_size,batch_size):
        #print(' -> '+str(i)+' / '+str(n))

        currObs=obsIDs[i:i+batch_size]

        obsIDs_S0=np.where(labels_train<0.5)[0]
        np.random.shuffle(obsIDs_S0)
        additionalObs=obsIDs_S0[:nb_add_obs_in_S0]

        X_curr_in=X_train_in[np.concatenate((currObs,additionalObs),axis=0),:].to(DEVICE)
        X_curr_out=X_train_out[np.concatenate((currObs,additionalObs),axis=0),:].to(DEVICE)

        labels_curr=labels_train[np.concatenate((currObs,additionalObs),axis=0),:].view(-1,1).to(DEVICE)

        # ===================forward=====================
        output0,output1,predS = model(X_curr_in)

        # ====================loss=======================
        
        #print(labels_curr)
        #print(criterion_S0(output0, X_curr_out))
        #print(criterion_S1(output1, X_curr_out))

        #loss=criterion_S0(output0, X_curr_out)+criterion_S1(output1, X_curr_out)
        
        lc=labels_curr.flatten() 
        
        mb_n1=float(0.01+lc.sum().to('cpu').detach())
        mb_n0=float(0.01+(1-lc).sum().to('cpu').detach())  #0.01 is to avoid dividing by zeros
        
        loss=torch.sum( torch.mean(torch.square(output0-X_curr_out),axis=1)*(1-lc)/mb_n0) +  torch.sum( torch.mean(torch.square(output1-X_curr_out),axis=1)*lc/mb_n1) + torch.mean(torch.square(predS-labels_curr))

        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        conv_losses_train.append(loss.item())


        # ======check the convergence on test data=======
        toto=np.arange(n_test)
        np.random.shuffle(toto)
        currObs=toto[:batch_size]

        X_curr=X_test[currObs,:].to(DEVICE)
        
        labels_curr=labels_test[currObs,:].view(-1,1).to(DEVICE)

        with torch.no_grad():
            output0,output1,predS = model(X_curr)
            

        lc=labels_curr.flatten()
        mb_n1=float(0.01+lc.sum().to('cpu').detach())
        mb_n0=float(0.01+(1-lc).sum().to('cpu').detach())  #0.01 is to avoid dividing by zeros
        
        loss=torch.sum( torch.mean(torch.square(output0-X_curr),axis=1)*(1-lc)/mb_n0)+torch.sum( torch.mean(torch.square(output1-X_curr),axis=1)*lc/mb_n1)+torch.mean(torch.square(predS-labels_curr))

        conv_losses_test.append(loss.item())

    #print('Current mini-batch losses:',conv_losses_train[-1],conv_losses_test[-1],)
    #print('Current MSE on pred S in train:',torch.mean(torch.square(predS-labels_curr)).item())


  return conv_losses_train,conv_losses_test


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# CLASS TO GENERATE CONTERFACTUAL TABULAR OBSERVATIONS
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



class CounterfactTablesGenerator:
    """
    A class to generate counterfactual tables in the case that there are two groups.
    """

    def __init__(self,ref_data,ref_S,cont_var,cat_var,latent_dim=3,DEVICE='cpu',num_epochs=100,batch_size=60,nb_add_obs_in_S0=4,learning_rate = 1e-3):
      """
      Initiate the class
      - ref_data: 2D numpy.array containing the data used to train the counterfactual generator
      - ref_S:  1D numpy.array containing the group in {0,1} of each observation in ref_data
      - cont_var: list mentioning which columns of ref_data represent continuous variables
      - cat_var: list mentioning which columns of ref_data represent categorical variables
      - latent_dim: is the dimension of the latent space of the S-informed AE to compute the counterfactuals
      - DEVICE: device for cuda (or not)

      Parameters related to how the S-informed AE is trained:
      - num_epochs=100
      - batch_size=60
      - nb_add_obs_in_S0=4  (the effective mini-batch size is batch_size+nb_add_obs_in_S0)
      - learning_rate = 1e-3

      General remarks:
      - Only the variables in cont_var and cat_var will be considered for the generation
        of conterfactual data
      - The variables in cat_var may be either binary or represent >2 groups.
      - When generating the counterfactual observations, the argmax principle will be used
        for all categorical variable (they are internally represented using ohe).
      """

      self.DEVICE=DEVICE

      #get and treat the data to train the S-aware AE parameters
      self.cont_var=cont_var
      self.cat_var=cat_var

      ref_data_cont=ref_data[:,self.cont_var]
      self.QuaTra = preprocessing.QuantileTransformer(n_quantiles=100)
      ref_data_cont_qt=self.QuaTra.fit_transform(ref_data_cont) #normalisation using quantile transform

      ref_data_cont_qt_NoXtremeVal=np.zeros([ref_data_cont_qt.shape[0],ref_data_cont_qt.shape[1]])
      ref_data_cont_qt_NoXtremeVal[:,:]=ref_data_cont_qt[:,:]
      locs=np.where(ref_data_cont_qt_NoXtremeVal<0.05)
      ref_data_cont_qt_NoXtremeVal[locs]=0.05
      locs=np.where(ref_data_cont_qt_NoXtremeVal>0.95)
      ref_data_cont_qt_NoXtremeVal[locs]=0.95       #the AE is slighly modified to avoid predicting extreme values

      ref_data_cat=ref_data[:,self.cat_var]
      self.OHEenc = preprocessing.OneHotEncoder()
      ref_data_cat_ohe=self.OHEenc.fit_transform(ref_data_cat).toarray()  #one-hot-encoding

      ref_data_4_ae_in=np.concatenate([ref_data_cont_qt,ref_data_cat_ohe],axis=1)
      ref_data_4_ae_out=np.concatenate([ref_data_cont_qt_NoXtremeVal,ref_data_cat_ohe],axis=1)

      #define the training and test observations, so that their are well balanced wrt S
      ref_S=ref_S.flatten()
      idS0=np.where(ref_S<0.5)[0]
      idS1=np.where(ref_S>=0.5)[0]

      print('There are',len(idS0),'observations in group 0 to train the generator')
      print('There are',len(idS1),'observations in group 1 to train the generator')

      tmp=np.arange(len(idS0))
      np.random.shuffle(tmp)
      thresh=int(3*len(idS0)/4)
      train_S0=idS0[tmp[:thresh]]
      test_S0=idS0[tmp[thresh:]]

      tmp=np.arange(len(idS1))
      np.random.shuffle(tmp)
      thresh=int(3*len(idS1)/4)
      train_S1=idS1[tmp[:thresh]]
      test_S1=idS1[tmp[thresh:]]

      train_data=np.concatenate([train_S0,train_S1],axis=0)  #merge the training data
      train_S=np.ones(train_data.shape[0])              #generate corresponding S labels
      train_S[:len(train_S0)]=0
      tmp=np.arange(train_data.shape[0])                #shuffle both
      np.random.shuffle(tmp)
      train_data=train_data[tmp]
      train_S=train_S[tmp]


      test_data=np.concatenate([test_S0,test_S1],axis=0)  #merge the test data
      test_S=np.ones(test_data.shape[0])             #generate corresponding S labels
      test_S[:len(test_S0)]=0
      tmp=np.arange(test_data.shape[0])              #shuffle both
      np.random.shuffle(tmp)
      test_data=test_data[tmp]
      test_S=test_S[tmp]

      #instanciate and train the S-aware AE
      p=ref_data_4_ae_in.shape[1]
      self.S_AE_model = autoencoder_explicit_S_tabularData(p,latent_dim).to(self.DEVICE)

      X_train_in=torch.tensor(ref_data_4_ae_in[train_data,:]).float()
      X_train_out=torch.tensor(ref_data_4_ae_out[train_data,:]).float()
      self.X_test=torch.tensor(ref_data_4_ae_in[test_data,:]).float()
      labels_train=torch.tensor(train_S).float().view(-1,1)
      self.labels_test=torch.tensor(test_S).float().view(-1,1)

      print("Train the S-informed AE for counterfactual data generation")
      self.evo_losses_train,self.evo_losses_test=fit_AE_SensitiveImpact(self.S_AE_model,X_train_in,X_train_out,labels_train,self.X_test,self.labels_test,DEVICE,num_epochs,batch_size,nb_add_obs_in_S0=nb_add_obs_in_S0,learning_rate = learning_rate)
      print("Job done")

      self.S_AE_model_cpu = self.S_AE_model.to('cpu')



    def check_convergence_quality(self):
        """
        allows to check whether the S-informed AE was properly trained
        """

        #1) loss convergence
        print("\nloss convergence:")
        plt.semilogy(self.evo_losses_train,label='train')
        plt.semilogy(self.evo_losses_test,label='test')
        plt.legend()
        plt.show()

        #2) one by one predictive power on the different variables
        with torch.no_grad():
            output_S0,output_S1,pred_S = self.S_AE_model_cpu(self.X_test)
        
        output=output_S0*(1-self.labels_test) + output_S1*self.labels_test

        Obs_S0=np.where(self.labels_test<0.5)[0]
        Obs_S1=np.where(self.labels_test>0.5)[0]
        print("\nNumber of test observations:")
        print(len(Obs_S0),'observations with S=0')
        print(len(Obs_S1),'observations with S=1')

        print("\nTrue vs predicted:")
        for i in range(output.shape[1]):
            pred=output[Obs_S0,i].detach().numpy()
            true=self.X_test[Obs_S0,i].detach().numpy()
            MAE=np.round(np.mean(np.abs(pred-true)),3)
            plt.scatter(pred,true,color='red',alpha=0.05)
            plt.plot([0.,1.],[0.,1.],'--')
            plt.title('G0: transformed variable '+str(i)+'  - MAE='+str(MAE))
            plt.show()

        for i in range(output.shape[1]):
            pred=output[Obs_S1,i].detach().numpy()
            true=self.X_test[Obs_S1,i].detach().numpy()
            MAE=np.round(np.mean(np.abs(pred-true)),3)
            plt.scatter(pred,true,color='blue',alpha=0.05)
            plt.plot([0.,1.],[0.,1.],'--')
            plt.title('G1: transformed variable '+str(i)+'  - MAE='+str(MAE))
            plt.show()


    def generate_counterfactuals(self,obs,S,no_cf_actually=False):
        """
        Generate the counterfactual observations of 'obs', with binary labels 'S'
        -> obs: observations for which the counterfactuals will be computed
        -> S: corresponding group in {0,1}
        -> no_cf_actually: non-counterfactual reconstruction if True (default=False)
        """

        #inverse the labels of S
        count_S=1-S

        if no_cf_actually:
            count_S[:]=S[:]  #to check what would be the non-counterfactual reconstruction

        #transform obs to be coherent with the S-informed AE
        obs_cont=obs[:,self.cont_var]
        obs_cont_qt=self.QuaTra.transform(obs_cont) #normalisation using quantile transform

        obs_cat=obs[:,self.cat_var]
        obs_cat_ohe=self.OHEenc.transform(obs_cat).toarray()  #one-hot-encoding

        obs_4_ae=np.concatenate([obs_cont_qt,obs_cat_ohe],axis=1)

        #predict the counterfactual on the transformed data
        obs_4_ae=torch.tensor(obs_4_ae).float()
        count_S=torch.tensor(count_S).float().view(-1,1)

        with torch.no_grad():
            output_S0,output_S1,predS = self.S_AE_model_cpu(obs_4_ae)
            
        output=output_S0*(1-count_S) + output_S1*count_S


        #transform output back to the original data format
        counter_obs_cont=self.QuaTra.inverse_transform(output[:,:len(self.cont_var)].numpy())
        counter_obs_cat=self.OHEenc.inverse_transform(output[:,len(self.cont_var):].numpy())

        #create the counterfactuals
        couter_obs=np.zeros([obs.shape[0],obs.shape[1]])
        couter_obs[:,:]=obs[:,:]
        couter_obs[:,self.cont_var]=counter_obs_cont
        couter_obs[:,self.cat_var]=counter_obs_cat

        return couter_obs





class CounterfactDataframeGenerator:
    """
    Wrapper class to use CounterfactTablesGenerator on pandas dataframes. All methods are the
    same as in CounterfactTablesGenerator. The two differences are:
     - the input and output data are in pandas dataframes instead of numpy arrays
     - the binary values for S are in 'col_S'
    """
    def __init__(self,ref_df,col_S,cont_var,cat_var,latent_dim=3,DEVICE='cpu',num_epochs=100,batch_size=60,nb_add_obs_in_S0=4,learning_rate = 1e-3):
        """
        see __init__ of CounterfactTablesGenerator and replace:
         - ref_data with ref_df
         - ref_S with col_S
        """
        self.cont_var=cont_var
        self.cat_var=cat_var
        self.col_S=col_S

        #manage the continuous and categorical variables
        if len(self.cat_var)>0:
            #get the categorical data (which can be integer or strings) by converting the them into integers
            #for numpy - there will have a problem in the QuaTra transforms of self.wrapped_CTG otherwise
            data_to_treat=np.zeros([ref_df[self.cat_var].shape[0],ref_df[self.cat_var].shape[1]])
            self.converters_int2label={}
            for i in range(len(self.cat_var)):
                self.converters_int2label[self.cat_var[i]]=list(set(ref_df[self.cat_var[i]]))
                groups=self.converters_int2label[self.cat_var[i]]
                ref_np_var=ref_df[self.cat_var[i]].to_numpy()
                for j in range(len(groups)):
                    tmp_loc=np.where(ref_np_var==groups[j])[0]
                    data_to_treat[tmp_loc,i]=j

        if len(self.cont_var)>0:
            if len(self.cat_var)==0:
                data_to_treat=ref_df[self.cont_var].to_numpy()
            else:
                data_cont=ref_df[self.cont_var].to_numpy()
                data_to_treat=np.concatenate([data_cont,data_to_treat],axis=1)

        cont_var_4_d2t=np.arange(0,len(self.cont_var))
        cat_var_4_d2t=np.arange(len(self.cont_var),len(self.cont_var)+len(self.cat_var))

        #manage the column S
        raw_S=ref_df[[self.col_S]].to_numpy().flatten()
        sensitive_values=list(set(raw_S))
        print("In S, 0 is for "+str(sensitive_values[0])+" and 1 is for "+str(sensitive_values[1]))

        S=np.zeros(ref_df.shape[0])
        S[np.where(raw_S==sensitive_values[1])[0]]=1
        S=S.reshape(-1,1)


        #create the wrapped CounterfactTablesGenerator
        self.wrapped_CTG=CounterfactTablesGenerator(data_to_treat,S,cont_var_4_d2t,cat_var_4_d2t,latent_dim=latent_dim,
                                               DEVICE=DEVICE,num_epochs=num_epochs,batch_size=batch_size,
                                               nb_add_obs_in_S0=nb_add_obs_in_S0,learning_rate = learning_rate)

    def check_convergence_quality(self):
        self.wrapped_CTG.check_convergence_quality()


    def generate_counterfactuals(self,df_obs,no_cf_actually=False):
        """
        Same a generate_counterfactuals of CounterfactTablesGenerator, except:
            - replace obs with df_obs
            - no need for S as we already know where to find it in the dataframe
        """

        #1) change the format of the input df

        #manage the continuous and categorical variables
        if len(self.cat_var)>0:
            #get the categorical data (which can be integer or strings) by converting the them into integers
            #for numpy - there will have a problem in the QuaTra transforms of self.wrapped_CTG otherwise
            data_to_treat=np.zeros([df_obs[self.cat_var].shape[0],df_obs[self.cat_var].shape[1]])
            for i in range(len(self.cat_var)):
                groups=self.converters_int2label[self.cat_var[i]]
                ref_np_var=df_obs[self.cat_var[i]].to_numpy()
                for j in range(len(groups)):
                    tmp_loc=np.where(ref_np_var==groups[j])[0]
                    data_to_treat[tmp_loc,i]=j

        if len(self.cont_var)>0:
            if len(self.cat_var)==0:
                data_to_treat=df_obs[self.cont_var].to_numpy()
            else:
                data_cont=df_obs[self.cont_var].to_numpy()
                data_to_treat=np.concatenate([data_cont,data_to_treat],axis=1)

        cont_var_4_d2t=np.arange(0,len(self.cont_var))
        cat_var_4_d2t=np.arange(len(self.cont_var),len(self.cont_var)+len(self.cat_var))

        #manage the column S
        raw_S=df_obs[[self.col_S]].to_numpy().flatten()
        sensitive_values=list(set(raw_S))
        #print("For the counterfactuals -> S0 refers to "+str(sensitive_values[0])+" and S1 refers to "+str(sensitive_values[1]))

        S=np.zeros(df_obs.shape[0])
        S[np.where(raw_S==sensitive_values[1])[0]]=1
        S=S.reshape(-1,1)

        #2) generate the counterfactuals
        couter_obs=self.wrapped_CTG.generate_counterfactuals(data_to_treat,S,no_cf_actually=no_cf_actually)


        #3) copy the result in the pertinent columns of the output dataframe

        #... init
        df_counter=df_obs.copy()
        
        list_IDs=list(df_counter.index)

        #... continuous variables
        for i in range(len(self.cont_var)):
            for loc in range(df_counter.shape[0]):
                df_counter[self.cont_var[i]][list_IDs[loc]]=couter_obs[loc,cont_var_4_d2t[i]]  #there might exist a parallel solution

        #... categorical variables
        for i in range(len(self.cat_var)):
            groups=self.converters_int2label[self.cat_var[i]]
            ref_np_var=couter_obs[:,cat_var_4_d2t[i]]
            for j in range(len(groups)):
                tmp_loc=np.where(ref_np_var==j)[0]
                for loc in tmp_loc:  #there might exist a parallel solution
                    df_counter[self.cat_var[i]][list_IDs[loc]]=groups[j]

        #... S
        for loc in range(df_counter.shape[0]):
            if df_counter[self.col_S][list_IDs[loc]]==sensitive_values[0]:
                df_counter[self.col_S][list_IDs[loc]]=sensitive_values[1]
            else:
                df_counter[self.col_S][list_IDs[loc]]=sensitive_values[0]


        return df_counter
