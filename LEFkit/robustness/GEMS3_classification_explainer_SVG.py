
from LEFkit.robustness.GEMS3_base_explainer import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale

class table_classif_explainer:
    """
    Class to analyze the behavior of a binary classifier using tabular data as input.
    """

    def __init__(self, X , y_pred, y_true=np.array([])):
        """
        Inputs:
            - X: 2D numpy array of the input observations. Each observation
            is on a row and each variable is on a column.
            - y_pred: vector numpy array of the binary predictions related to the
            observations of 'X'
            - y_true: (optional) vector numpy array having the same size as y_pred.
            It will make it possible to analyze the prediction errors.
        """
        self.X=X
        self.y_pred=y_pred
        if len(y_true)>0:
            self.y_true=y_true
            self.y_true_is_known=True
            self.errors=1*(np.abs(self.y_pred-self.y_true)>0.5)
            #print(self.y_pred)
            #print(self.y_true)
            #print(np.mean(np.abs(y_pred-y_true)))
            #print(self.errors.max())
        else:
            self.y_true_is_known=False

    def Get_X(self):
        return self.X

    def Get_y_pred(self):
        return self.y_pred

    def Get_y_true_is_known(self):
        return self.y_true_is_known

    def Get_y_true(self):
        return self.y_true


    def plot_mean_influence_on_pred(self, X_column_index,X_column_name='Null',y_axis_min_max=[0.,0.],plot_results=True,influence_cat_var={'Show':False},ListStressValues={'known':False},cpt_confidence_interval=False):
        """
            Plot the influence of a variable mean  in `self.X` on the binary predictions self.y_pred.

            Inputs:
                - X_column_index: Index of the column that will be studied in 'self.X'
                - X_column_name: Name of the variable corresponding to the column
                'X_column_index' in 'self.X'
                - y_axis_min_max: if defined, contains a list [y_axis_min,y_axis_max] with the min and max value on the y-axis in the plot
                - plot_results: set to False to avoid showing the influence plot (its values will be simply returned)
                - influence_cat_var: show the influence of a categorical variable if influence_cat_var['Show']==True. The column of this
                                     variable in self.X is influence_cat_var['Col']. A dictionary with the label of each of its possible
                                     values is influence_cat_var['DictNameVar'] (e.g. influence_cat_var['DictNameVar'][1]='Yes').
                - ListStressValues: define the values on which the data will be stressed if ListStressValues['known']==True. These value
                                    will be the deciles of self.X[:,X_column_index] otherwise. If defined, ListStressValues['listValues']
                                    must contain the stress values and ListStressValues['listScaledValues'] must contain the corresponding
                                    values if self.X[:,X_column_index] is centered-reduced.
        """

        #observations of the variable of interest
        input_obs=self.X[:,X_column_index].reshape(-1,1)
        input_obs_scaled=self.X[:,X_column_index].reshape(-1,1)
        input_obs_scaled = scale(input_obs_scaled, with_mean=True, with_std=True, copy=True )

        #extract the quantiles of interest
        if ListStressValues['known']:
            list_StressValues=ListStressValues['listValues']
            list_StressValues_scaled=ListStressValues['listScaledValues']
        else:
            q_1_9=np.quantile(input_obs, [0.1,0.9])
            n_q_1_9=np.quantile(input_obs_scaled, [0.1,0.9])
            list_StressValues=np.arange(q_1_9[0],q_1_9[1],(q_1_9[1]-q_1_9[0])/5.001)
            list_StressValues_scaled=np.arange(n_q_1_9[0],n_q_1_9[1],(n_q_1_9[1]-n_q_1_9[0])/5.001)

        #compute the impact of the stress
        #... init (general)
        list_res={'p':[],'stress_values':[],'weight_means':[]}
        obs_strs=obs_stresser(input_obs_scaled)

        if cpt_confidence_interval:
            nb_drawn_subsamples=80
            m_obs_strs=obs_stresser_multiple(input_obs_scaled,nb_drawn_subsamples,int(len(input_obs_scaled)*0.5))
            list_wm_CIs_q1=[]
            list_wm_CIs_q9=[]

        for i in range(len(list_StressValues_scaled)):
            #...estimation (general)
            #ksis=obs_strs.fit('mean', [0, list_StressValues_scaled[i]],gd_iterations=150)
            ksis=obs_strs.fit({'means': {0: list_StressValues_scaled[i]}},gd_iterations=150)
            lambdas=obs_strs.get_lambda()

            #...estimation (global sensitivity)
            weight_mean=np.average(self.y_pred.flatten(), weights=lambdas)
            list_res['stress_values'].append(list_StressValues[i])
            list_res['weight_means'].append(weight_mean)

            #optional confidence intervals
            if cpt_confidence_interval:
                #m_obs_strs.fit_and_cpt_lambdas('mean', [0, list_StressValues_scaled[i]],gd_iterations=150)
                m_obs_strs.fit_and_cpt_lambdas({'means': {0: list_StressValues_scaled[i]}},gd_iterations=150)
                LstQ1Q9=m_obs_strs.CptQuantilesOfWeightedSumsWithLambdas(self.y_pred.flatten(),[0.1,0.9])
                list_wm_CIs_q1.append(LstQ1Q9[0])
                list_wm_CIs_q9.append(LstQ1Q9[1])

        #plot results
        if plot_results:
            #... global sensitivity
            plt.plot(list_res['stress_values'],list_res['weight_means'],color='b')

            if cpt_confidence_interval:
                plt.fill_between(list_res['stress_values'], list_wm_CIs_q1, list_wm_CIs_q9, alpha=0.2,color='b')


            #... impact of categorical subgroups
            if influence_cat_var['Show']:
                icv_sub_lists_res_weight_means={}
                icv_sub_lists_res_quantiles={}
                icv_categ_name={}
                icv_categories=np.unique(self.X[:,influence_cat_var['Col']].astype(np.int))

                for category in icv_categories:
                    icv_categ_name[category]=influence_cat_var['DictNameVar'][category]
                    loc_obs_of_interest=np.where(self.X[:,influence_cat_var['Col']]==category)[0]
                    if len(loc_obs_of_interest)>10:
                        loc_explainer=table_classif_explainer(self.X[loc_obs_of_interest,:],self.y_pred[loc_obs_of_interest])
                        icv_sub_lists_res_quantiles[category],icv_sub_lists_res_weight_means[category]=loc_explainer.plot_mean_influence_on_pred(X_column_index,plot_results=False)

                        frequency_level=0.5*(len(loc_obs_of_interest)/self.X.shape[0])+0.5
                        curlabel=str(icv_categ_name[category])+' ('+str(len(loc_obs_of_interest))+' obs)'
                        plt.plot(icv_sub_lists_res_quantiles[category],icv_sub_lists_res_weight_means[category],'--',label=curlabel,alpha=frequency_level)
                    else:
                        print('Not enough observations in category '+str(icv_categ_name[category]))

                if len(icv_categories)<15:
                    plt.legend(fontsize=10)

            #...general plot properties
            if y_axis_min_max[0]!=y_axis_min_max[1]:
                plt.ylim(ymax = y_axis_min_max[1], ymin = y_axis_min_max[0])

            if X_column_name=='Null':
                plt.xlabel('Mean value')
            else:
                plt.xlabel('Mean of '+X_column_name)
            plt.ylabel('Portion predicted 1s')

            plt.show()

        return list_res['stress_values'],list_res['weight_means']

    def plot_mean_influence_on_DispImpact(self, X_column_index,S,X_column_name='Null',y_axis_min_max=[0.,0.],plot_results=True,influence_cat_var={'Show':False},cpt_confidence_interval=False):
        """
            Plot the influence of a variable mean in `self.X` on the disparate impact of the binary predictions self.y_pred.

            Inputs:
                - X_column_index: Index of the column that will be studied in 'self.X'
                - S: vector of binary values [0,1] representing the two groups in the observations of self.X and self.y_pred
                - X_column_name: Name of the variable corresponding to the column 'X_column_index' in 'self.X'
                - y_axis_min_max: if defined, contains a list [y_axis_min,y_axis_max] with the min and max value on the y-axis in the plot
                - influence_cat_var: show the influence of a categorical variable if influence_cat_var['Show']==True. The column of this
                                     variable in self.X is influence_cat_var['Col']. A dictionary with the label of each of its possible
                                     values is influence_cat_var['DictNameVar'] (e.g. influence_cat_var['DictNameVar'][1]='Yes').
        """
        #get the observation IDs in both groups
        Obs_G0=np.where(S<0.5)
        Obs_G1=np.where(S>=0.5)

        input_obs0=self.X[Obs_G0,X_column_index].reshape(-1,1)
        input_obs1=self.X[Obs_G1,X_column_index].reshape(-1,1)

        m0=input_obs0.mean()
        m1=input_obs1.mean()
        s0=input_obs0.std()
        s1=input_obs1.std()

        y_pred0=self.y_pred[Obs_G0].flatten()
        y_pred1=self.y_pred[Obs_G1].flatten()

        #generate a grid of stressed means between the first and last deciles of the two distributions
        tested_mean_min=np.max([np.quantile(input_obs0, 0.2),np.quantile(input_obs1, 0.2)])
        tested_mean_max=np.min([np.quantile(input_obs0, 0.8),np.quantile(input_obs1, 0.8)])
        tested_mean_step=(tested_mean_max-tested_mean_min)/5.001

        if tested_mean_step>0.:
            list_tested_means=np.arange(tested_mean_min,tested_mean_max,tested_mean_step)
        else:
            list_tested_means=np.array([tested_mean_min])

        #observation stressers in the two group
        obs_strs0=obs_stresser((input_obs0-m0)/s0)
        obs_strs1=obs_stresser((input_obs1-m1)/s1) #the stresser prefers centered-reduced data

        if cpt_confidence_interval:
            nb_drawn_subsamples=80
            m_obs_strs0=obs_stresser_multiple((input_obs0-m0)/s0,nb_drawn_subsamples,int(len(input_obs0)*0.5))
            m_obs_strs1=obs_stresser_multiple((input_obs1-m1)/s1,nb_drawn_subsamples,int(len(input_obs1)*0.5))


        #stress the data and compute the Disparate Impacts
        list_DIs=[]
        if cpt_confidence_interval:
            list_DI_CIs_q1=[]
            list_DI_CIs_q9=[]

        for tested_mean in list_tested_means:
            #ksis0=obs_strs0.fit('mean', [0, (tested_mean-m0)/s0],gd_iterations=150)
            ksis0=obs_strs0.fit({'means': {0: (tested_mean-m0)/s0}},gd_iterations=150)
            lambdas0=obs_strs0.get_lambda()
            weight_mean0=np.average(y_pred0, weights=lambdas0)  #remark: the sum of the weights is 1, so a weighted sum would give the same result

            #ksis1=obs_strs1.fit('mean', [0, (tested_mean-m1)/s1],gd_iterations=150)
            ksis1=obs_strs1.fit({'means': {0: (tested_mean-m1)/s1}},gd_iterations=150)
            lambdas1=obs_strs1.get_lambda()
            weight_mean1=np.average(y_pred1, weights=lambdas1)  #remark: the sum of the weights is 1, so a weighted sum would give the same result

            #print(weight_mean0,weight_mean1)

            list_DIs.append(weight_mean0/weight_mean1)

            if cpt_confidence_interval:
                #m_obs_strs0.fit_and_cpt_lambdas('mean', [0, (tested_mean-m0)/s0],gd_iterations=150)
                m_obs_strs0.fit_and_cpt_lambdas({'means': {0: (tested_mean-m0)/s0}},gd_iterations=150)
                #m_obs_strs1.fit_and_cpt_lambdas('mean', [0, (tested_mean-m1)/s1],gd_iterations=150)
                m_obs_strs1.fit_and_cpt_lambdas({'means': {0: (tested_mean-m1)/s1}},gd_iterations=150)

                WeightedSums0=m_obs_strs0.CptWeightedSumsWithLambdas(y_pred0)
                WeightedSums1=m_obs_strs1.CptWeightedSumsWithLambdas(y_pred1)

                DI_samples=np.zeros(nb_drawn_subsamples)

                for i in range(nb_drawn_subsamples):
                    DI_samples[i]=WeightedSums0[i]/WeightedSums1[i]

                loc_quantiles=np.quantile(DI_samples,[0.1,0.9])

                list_DI_CIs_q1.append(loc_quantiles[0])
                list_DI_CIs_q9.append(loc_quantiles[1])


        #plot results
        if plot_results:
            #... global sensitivity
            #print(list_tested_means)
            #print(list_DIs)
            plt.plot(list_tested_means,list_DIs,'b')

            if cpt_confidence_interval:
                plt.fill_between(list_tested_means, list_DI_CIs_q1, list_DI_CIs_q9, alpha=0.2,color='b')


            #... impact of categorical subgroups
            if influence_cat_var['Show']:
                icv_sub_lists_x={}
                icv_sub_lists_y={}
                icv_categ_name={}
                icv_categories=np.unique(self.X[:,influence_cat_var['Col']].astype(np.int))

                for category in icv_categories:
                    icv_categ_name[category]=influence_cat_var['DictNameVar'][category]
                    loc_obs_of_interest=np.where(self.X[:,influence_cat_var['Col']]==category)[0]
                    loc_explainer=table_classif_explainer(self.X[loc_obs_of_interest,:],self.y_pred[loc_obs_of_interest])

                    icv_sub_lists_x[category],icv_sub_lists_y[category]=loc_explainer.plot_mean_influence_on_DispImpact(X_column_index,S[loc_obs_of_interest],plot_results=False)

                    frequency_level=0.5*(len(loc_obs_of_interest)/self.X.shape[0])+0.5
                    curlabel=str(icv_categ_name[category])+' ('+str(len(loc_obs_of_interest))+' obs)'
                    plt.plot(icv_sub_lists_x[category],icv_sub_lists_y[category],'--',label=curlabel,alpha=frequency_level)


                if len(icv_categories)<15:
                    plt.legend(fontsize=10)

            #...general plot properties
            if y_axis_min_max[0]!=y_axis_min_max[1]:
                plt.ylim(ymax = y_axis_min_max[1], ymin = y_axis_min_max[0])

            if plt.gca().get_ylim()[0]>0.8:
                plt.ylim(ymin = 0.8)

            if plt.gca().get_ylim()[1]<1.2:
                plt.ylim(ymax = 1.2)

            if X_column_name=='Null':
                plt.xlabel('Mean value')
            else:
                plt.xlabel('Mean of '+X_column_name)
            plt.ylabel('Disparate Impact')

            plt.show()

        if cpt_confidence_interval:
            return list_tested_means,list_DIs,list_DI_CIs_q1,list_DI_CIs_q9
        else:
            return list_tested_means,list_DIs



    def plot_independent_mean_influences_on_pred(self, X_column_indices,X_column_names='Null',y_axis_min_max=[0.,0.]):
        """
            Plot the +independent+ influence of several variable mean  in `self.X` on the binary predictions self.y_pred.

            Inputs:
                - X_column_indices: List of the column indices that will be independently studied in 'self.X'
                - X_column_names: List of the variable names in the corresponding columns
                'X_column_index' in 'self.X'
                - y_axis_min_max: if defined, contains a list [y_axis_min,y_axis_max] with the min and max value on the y-axis in the plot
        """

        font = {'family' : 'normal',
                    'weight' : 'bold',
                    'size'   : 11}#22
        plt.rc('font', **font)

        for i in range(len(X_column_indices)):
            X_column_index=X_column_indices[i]
            if X_column_names=='Null':
                X_column_name='X'+str(i+1)
            else:
                X_column_name=X_column_names[i]

            #observations of the variable of interest
            input_obs=self.X[:,X_column_index].reshape(-1,1)
            input_obs_scaled=self.X[:,X_column_index].reshape(-1,1)
            input_obs_scaled = scale(input_obs_scaled, with_mean=True, with_std=True, copy=True )

            #extract the quantiles of interest
            q_1_9=np.quantile(input_obs, [0.1,0.9])
            n_q_1_9=np.quantile(input_obs_scaled, [0.1,0.9])
            list_StressValues=np.arange(q_1_9[0],q_1_9[1],(q_1_9[1]-q_1_9[0])/5.001)
            list_StressValues_scaled=np.arange(n_q_1_9[0],n_q_1_9[1],(n_q_1_9[1]-n_q_1_9[0])/5.001)

            #compute the impact of the stress
            list_res={'p':[],'stress_values':[],'weight_means':[]}
            obs_strs=obs_stresser(input_obs_scaled)

            for i in range(len(list_StressValues_scaled)):
                #ksis=obs_strs.fit('mean', [0, list_StressValues_scaled[i]],gd_iterations=150)
                ksis=obs_strs.fit({'means': {0: list_StressValues_scaled[i]}},gd_iterations=150)
                lambdas=obs_strs.get_lambda()
                weight_mean=np.average(self.y_pred.flatten(), weights=lambdas)
                #list_res['stress_values'].append(list_StressValues[i])
                list_res['stress_values'].append(list_StressValues_scaled[i])
                list_res['weight_means'].append(weight_mean)

            plt.plot(list_res['stress_values'],list_res['weight_means'], label=X_column_name)

        if y_axis_min_max[0]!=y_axis_min_max[1]:
            plt.ylim(ymax = y_axis_min_max[1], ymin = y_axis_min_max[0])

        #plt.rc('xtick', labelsize=22)
        #plt.rc('ytick', labelsize=22)
        plt.rc('xtick', labelsize=11)
        plt.rc('ytick', labelsize=11)

        #plt.xlabel('Mean value')
        plt.xlabel('Stress level on the mean')
        plt.ylabel('Portion predicted 1s')
        plt.legend(fontsize=10)
        plt.show()


    def plot_mean_influence_on_errors(self, X_column_index,X_column_name='Null',y_axis_min_max=[0.,0.]):
        """
            Plot the influence of a variable mean  in `self.X` on the errors made on the binary predictions self.y_pred (compared with self.y_true).

            Inputs:
                - X_column_index: Index of the column that will be studied in 'self.X'
                - X_column_name: Name of the variable corresponding to the column
                'X_column_index' in 'self.X'
                - y_axis_min_max: if defined, contains a list [y_axis_min,y_axis_max] with the min and max value on the y-axis in the plot
        """

        #observations of the variable of interest
        input_obs=self.X[:,X_column_index].reshape(-1,1)
        input_obs_scaled=self.X[:,X_column_index].reshape(-1,1)
        input_obs_scaled = scale(input_obs_scaled, with_mean=True, with_std=True, copy=True )

        #extract the quantiles of interest
        q_1_9=np.quantile(input_obs, [0.1,0.9])
        n_q_1_9=np.quantile(input_obs_scaled, [0.1,0.9])
        list_StressValues=np.arange(q_1_9[0],q_1_9[1],(q_1_9[1]-q_1_9[0])/5.001)
        list_StressValues_scaled=np.arange(n_q_1_9[0],n_q_1_9[1],(n_q_1_9[1]-n_q_1_9[0])/5.001)

        #compute the impact of the stress
        list_res={'p':[],'stress_values':[],'weight_means':[]}
        obs_strs=obs_stresser(input_obs_scaled)

        for i in range(len(list_StressValues_scaled)):
            #ksis=obs_strs.fit('mean', [0, list_StressValues_scaled[i]],gd_iterations=150)
            ksis=obs_strs.fit({'means': {0: list_StressValues_scaled[i]}},gd_iterations=150)
            lambdas=obs_strs.get_lambda()
            weight_mean=np.average(self.errors.flatten(), weights=lambdas)
            list_res['stress_values'].append(list_StressValues[i])
            list_res['weight_means'].append(weight_mean)

        plt.plot(list_res['stress_values'],list_res['weight_means'])
        if X_column_name=='Null':
            plt.xlabel('Mean value')
        else:
            plt.xlabel(X_column_name)
        plt.ylabel('Portion of errors')

        if y_axis_min_max[0]!=y_axis_min_max[1]:
            plt.ylim(ymax = y_axis_min_max[1], ymin = y_axis_min_max[0])

        plt.show()


    def plot_two_mean_influences_on_pred(self, X_column_index_1, X_column_index_2,X_column_name_1='Null',X_column_name_2='Null'):
        """
            Plot the influence of two variable means in `self.X` on the binary predictions self.y_pred.

            Inputs:
                - X_column_index_1: Index of the first studied column in 'self.X'
                - X_column_index_2: Index of the second studied column in 'self.X'
                - X_column_name_1: Name of the variable represented in column X_column_index_1
                - X_column_name_2: Name of the variable represented in column X_column_index_2
        """

        #observations of the variable of interest
        input_obs_1=self.X[:,X_column_index_1].reshape(-1,1)
        input_obs_1_scaled=self.X[:,X_column_index_1].reshape(-1,1)
        input_obs_1_scaled = scale(input_obs_1_scaled, with_mean=True, with_std=True, copy=True )

        input_obs_2=self.X[:,X_column_index_2].reshape(-1,1)
        input_obs_2_scaled=self.X[:,X_column_index_2].reshape(-1,1)
        input_obs_2_scaled = scale(input_obs_2_scaled, with_mean=True, with_std=True, copy=True )

        input_obs_scaled=np.concatenate((input_obs_1_scaled,input_obs_2_scaled),axis=1)

        #extract the quantiles of interest
        quantile_1_p01=np.quantile(input_obs_1, 0.1)
        quantile_1_p09=np.quantile(input_obs_1, 0.9)
        list_val_1=np.arange(quantile_1_p01,quantile_1_p09,(quantile_1_p09-quantile_1_p01)/5.01)
        quantile_scaled_1_p01=np.quantile(input_obs_1_scaled, 0.2)
        quantile_scaled_1_p09=np.quantile(input_obs_1_scaled, 0.8)
        list_val_scaled_1=np.arange(quantile_scaled_1_p01,quantile_scaled_1_p09,(quantile_scaled_1_p09-quantile_scaled_1_p01)/5.01)

        quantile_2_p01=np.quantile(input_obs_2, 0.2)
        quantile_2_p09=np.quantile(input_obs_2, 0.8)
        list_val_2=np.arange(quantile_2_p01,quantile_2_p09,(quantile_2_p09-quantile_2_p01)/5.01)
        quantile_scaled_2_p01=np.quantile(input_obs_2_scaled, 0.2)
        quantile_scaled_2_p09=np.quantile(input_obs_2_scaled, 0.8)
        list_val_scaled_2=np.arange(quantile_scaled_2_p01,quantile_scaled_2_p09,(quantile_scaled_2_p09-quantile_scaled_2_p01)/5.01)

        #compute the impact of the stress
        mat_res=np.zeros([len(list_val_1),len(list_val_2)])

        obs_strs=obs_stresser(input_obs_scaled)

        for i in range(len(list_val_scaled_1)):
            for j in range(len(list_val_scaled_2)):
                #ksis=obs_strs.fit('means', [0, list_val_scaled_1[i],1, list_val_scaled_2[j]],gd_iterations=200)
                ksis=obs_strs.fit({'means': {0: list_val_scaled_1[i],1: list_val_scaled_2[j]}},gd_iterations=200)
                lambdas=obs_strs.get_lambda()
                weight_mean=np.average(self.y_pred.flatten(), weights=lambdas)
                mat_res[i,j]=weight_mean

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(mat_res)
        fig.colorbar(cax)
        xtl=['']
        for val in list_val_1:
            xtl.append(str(np.round(val,1)))
        ytl=['']
        for val in list_val_2:
            ytl.append(str(np.round(val,1)))
        ax.set_xticklabels(xtl)
        ax.set_yticklabels(ytl)
        if X_column_name_1=='Null':
            plt.xlabel('Mean of variable 1')
        else:
            plt.xlabel('Mean '+X_column_name_1)
        if X_column_name_2=='Null':
            plt.ylabel('Mean of variable 2')
        else:
            plt.ylabel('Mean '+X_column_name_2)
        plt.title('Portion of predicted 1s')
        plt.show()

    def plot_two_mean_influences_on_errors(self, X_column_index_1, X_column_index_2,X_column_name_1='Null',X_column_name_2='Null'):
        """
            Plot the influence of two variable means in `self.X` on the errors made on the binary predictions self.y_pred (compared with self.y_true).

            Inputs:
                - X_column_index_1: Index of the first studied column in 'self.X'
                - X_column_index_2: Index of the second studied column in 'self.X'
                - X_column_name_1: Name of the variable represented in column X_column_index_1
                - X_column_name_2: Name of the variable represented in column X_column_index_2
        """

        #observations of the variable of interest
        input_obs_1=self.X[:,X_column_index_1].reshape(-1,1)
        input_obs_1_scaled=self.X[:,X_column_index_1].reshape(-1,1)
        input_obs_1_scaled = scale(input_obs_1_scaled, with_mean=True, with_std=True, copy=True )

        input_obs_2=self.X[:,X_column_index_2].reshape(-1,1)
        input_obs_2_scaled=self.X[:,X_column_index_2].reshape(-1,1)
        input_obs_2_scaled = scale(input_obs_2_scaled, with_mean=True, with_std=True, copy=True )

        input_obs_scaled=np.concatenate((input_obs_1_scaled,input_obs_2_scaled),axis=1)

        #extract the quantiles of interest
        quantile_1_p01=np.quantile(input_obs_1, 0.1)
        quantile_1_p09=np.quantile(input_obs_1, 0.9)
        list_val_1=np.arange(quantile_1_p01,quantile_1_p09,(quantile_1_p09-quantile_1_p01)/5.01)
        quantile_scaled_1_p01=np.quantile(input_obs_1_scaled, 0.2)
        quantile_scaled_1_p09=np.quantile(input_obs_1_scaled, 0.8)
        list_val_scaled_1=np.arange(quantile_scaled_1_p01,quantile_scaled_1_p09,(quantile_scaled_1_p09-quantile_scaled_1_p01)/5.01)

        quantile_2_p01=np.quantile(input_obs_2, 0.2)
        quantile_2_p09=np.quantile(input_obs_2, 0.8)
        list_val_2=np.arange(quantile_2_p01,quantile_2_p09,(quantile_2_p09-quantile_2_p01)/5.01)
        quantile_scaled_2_p01=np.quantile(input_obs_2_scaled, 0.2)
        quantile_scaled_2_p09=np.quantile(input_obs_2_scaled, 0.8)
        list_val_scaled_2=np.arange(quantile_scaled_2_p01,quantile_scaled_2_p09,(quantile_scaled_2_p09-quantile_scaled_2_p01)/5.01)


        #compute the impact of the stress
        mat_res=np.zeros([len(list_val_1),len(list_val_2)])
        mat_TPR=np.zeros([len(list_val_1),len(list_val_2)])
        mat_TNR=np.zeros([len(list_val_1),len(list_val_2)])

        obs_strs=obs_stresser(input_obs_scaled)

        for i in range(len(list_val_scaled_1)):
            for j in range(len(list_val_scaled_2)):
                #ksis=obs_strs.fit('means', [0, list_val_scaled_1[i],1, list_val_scaled_2[j]],gd_iterations=200)
                ksis=obs_strs.fit({'means': {0: list_val_scaled_1[i],1: list_val_scaled_2[j]}},gd_iterations=200)
                lambdas=obs_strs.get_lambda()
                weight_mean=np.average(self.errors.flatten(), weights=lambdas)
                mat_res[i,j]=weight_mean
                #additionally compute the True positive rate (true positive prediction over true predictions) and the true negative rate
                wnb_pred1=np.average(self.y_pred.flatten(), weights=lambdas)
                wnb_pred0=np.average(1.-self.y_pred.flatten(), weights=lambdas)
                TP=(self.y_true.flatten())*(self.y_pred.flatten())
                wnb_TP=np.average(TP, weights=lambdas)
                TN=(1.-self.y_true.flatten())*(1.-self.y_pred.flatten())
                wnb_TN=np.average(TN, weights=lambdas)
                mat_TPR[i,j]=wnb_TP/wnb_pred1
                mat_TNR[i,j]=wnb_TN/wnb_pred0

        for shown_info in ['Portion of errors','True positive rate','True negative rate']:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            if shown_info=='Portion of errors':
                cax = ax.matshow(mat_res)
            elif shown_info=='True positive rate':
                cax = ax.matshow(mat_TPR)
            else:
                cax = ax.matshow(mat_TNR)
            fig.colorbar(cax)
            xtl=['']
            for val in list_val_1:
                xtl.append(str(np.round(val,1)))
            ytl=['']
            for val in list_val_2:
                ytl.append(str(np.round(val,1)))
            ax.set_xticklabels(xtl)
            ax.set_yticklabels(ytl)
            if X_column_name_1=='Null':
                plt.xlabel('Mean of variable 1')
            else:
                plt.xlabel('Mean '+X_column_name_1)
            if X_column_name_2=='Null':
                plt.ylabel('Mean of variable 2')
            else:
                plt.ylabel('Mean '+X_column_name_2)
            plt.title(shown_info)
            plt.show()





    def plot_std_influence_on_pred(self, X_column_index,X_column_name='Null',y_axis_min_max=[0.,0.]):
        """
            Plot the influence of a variable standard deviation  in `self.X` on the binary predictions self.y_pred.

            Inputs:
                - X_column_index: Index of the column that will be studied in 'self.X'
                - X_column_name: Name of the variable corresponding to the column
                'X_column_index' in 'self.X'
                - y_axis_min_max: if defined, contains a list [y_axis_min,y_axis_max] with the min and max value on the y-axis in the plot
        """

        #observations of the variable of interest
        input_obs=self.X[:,X_column_index].reshape(-1,1)
        input_obs_mean=input_obs.mean()
        input_obs_std=input_obs.std()
        input_obs_scaled=(input_obs-input_obs_mean)/input_obs_std

        #extract the quantiles of interest
        list_std=np.arange(0.5,2.0001,0.2)

        #compute the impact of the stress
        list_res={'std':[],'weight_means':[]}
        obs_strs=obs_stresser(input_obs_scaled)

        for i in range(len(list_std)):
            #ksis=obs_strs.fit('variance', [0, list_std[i]*list_std[i]],gd_iterations=150)
            ksis=obs_strs.fit({'var': [0, list_std[i]*list_std[i]]},gd_iterations=150)
            lambdas=obs_strs.get_lambda()
            weight_mean=np.average(self.y_pred.flatten(), weights=lambdas)
            list_res['std'].append(list_std[i]*input_obs_std)
            list_res['weight_means'].append(weight_mean)

        plt.plot(list_res['std'],list_res['weight_means'])
        if X_column_name=='Null':
            plt.xlabel('std')
        else:
            plt.xlabel('std of '+X_column_name+' (mean='+str(np.round(input_obs_mean,2))+')')
        plt.ylabel('Portion predicted 1s')

        if y_axis_min_max[0]!=y_axis_min_max[1]:
            plt.ylim(ymax = y_axis_min_max[1], ymin = y_axis_min_max[0])

        plt.show()

    def plot_std_influence_on_errors(self, X_column_index,X_column_name='Null',y_axis_min_max=[0.,0.]):
        """
            Plot the influence of a variable standard deviation  in `self.X` on the errors made on the binary predictions self.y_pred (compared with self.y_true).

            Inputs:
                - X_column_index: Index of the column that will be studied in 'self.X'
                - X_column_name: Name of the variable corresponding to the column
                'X_column_index' in 'self.X'
                - y_axis_min_max: if defined, contains a list [y_axis_min,y_axis_max] with the min and max value on the y-axis in the plot
        """

        #observations of the variable of interest
        input_obs=self.X[:,X_column_index].reshape(-1,1)
        input_obs_mean=input_obs.mean()
        input_obs_std=input_obs.std()
        input_obs_scaled=(input_obs-input_obs_mean)/input_obs_std

        #extract the quantiles of interest
        list_std=np.arange(0.5,2.0001,0.2)

        #compute the impact of the stress
        list_res={'std':[],'weight_means':[]}
        obs_strs=obs_stresser(input_obs_scaled)

        for i in range(len(list_std)):
            #ksis=obs_strs.fit('variance', [0, list_std[i]*list_std[i]],gd_iterations=150)
            ksis=obs_strs.fit({'var': [0, list_std[i]*list_std[i]]},gd_iterations=150)
            lambdas=obs_strs.get_lambda()
            weight_mean=np.average(self.errors.flatten(), weights=lambdas)
            list_res['std'].append(list_std[i]*input_obs_std)
            list_res['weight_means'].append(weight_mean)

        plt.plot(list_res['std'],list_res['weight_means'])
        if X_column_name=='Null':
            plt.xlabel('std')
        else:
            plt.xlabel('std of '+X_column_name+' (mean='+str(np.round(input_obs_mean,2))+')')
        plt.ylabel('Portion of errors')

        if y_axis_min_max[0]!=y_axis_min_max[1]:
            plt.ylim(ymax = y_axis_min_max[1], ymin = y_axis_min_max[0])

        plt.show()



    def plot_correlation_influence_on_pred(self, X_column_index_1, X_column_index_2,X_column_name_1='Null',X_column_name_2='Null',y_axis_min_max=[0.,0.]):
        """
            Plot the influence of the correlation between two variables in `self.X` on the binary predictions self.y_pred.

            Inputs:
                - X_column_index_1: Index of the first studied column in 'self.X'
                - X_column_index_2: Index of the second studied column in 'self.X'
                - X_column_name_1: Name of the variable represented in column X_column_index_1
                - X_column_name_2: Name of the variable represented in column X_column_index_2
                - y_axis_min_max: if defined, contains a list [y_axis_min,y_axis_max] with the min and max value on the y-axis in the plot
        """

        #observations of the variable of interest
        input_obs1=self.X[:,X_column_index_1].reshape(-1,1)
        input_obs1_mean=input_obs1.mean()
        input_obs1_std=input_obs1.std()
        input_obs1_scaled=(input_obs1-input_obs1_mean)/input_obs1_std

        input_obs2=self.X[:,X_column_index_2].reshape(-1,1)
        input_obs2_mean=input_obs2.mean()
        input_obs2_std=input_obs2.std()
        input_obs2_scaled=(input_obs2-input_obs2_mean)/input_obs2_std

        input_obs_scaled=np.concatenate((input_obs1_scaled,input_obs2_scaled),axis=1)

        corr_actual =  (input_obs1_scaled*input_obs2_scaled).sum()/(len(input_obs1_scaled)-1.)
        corr_plus=corr_actual+0.4
        if corr_plus>1.:
            corr_plus=1.
        corr_minus=corr_actual-0.4
        if corr_minus<-1.:
            corr_minus=-1.

        #extract the quantiles of interest
        list_corr=np.arange(corr_minus,corr_plus,0.07)

        #compute the impact of the stress
        list_res={'corr':[],'weight_means':[]}
        obs_strs=obs_stresser(input_obs_scaled)

        for i in range(len(list_corr)):
            #ksis=obs_strs.fit('covariance', [0, 1,list_corr[i]],gd_iterations=150)
            ksis=obs_strs.fit({'cov': [0, 1,list_corr[i]]},gd_iterations=150)
            lambdas=obs_strs.get_lambda()
            weight_mean=np.average(self.y_pred.flatten(), weights=lambdas)
            list_res['corr'].append(list_corr[i])
            list_res['weight_means'].append(weight_mean)

        #print(lambdas)

        plt.plot(list_res['corr'],list_res['weight_means'])
        if X_column_name_1=='Null' or X_column_name_2=='Null':
            plt.xlabel('Correlation')
        else:
            plt.xlabel('Correlation between '+X_column_name_1+' and '+X_column_name_2)
        plt.ylabel('Portion predicted 1s')

        if y_axis_min_max[0]!=y_axis_min_max[1]:
            plt.ylim(ymax = y_axis_min_max[1], ymin = y_axis_min_max[0])

        plt.show()

    def plot_correlation_influence_on_errors(self, X_column_index_1, X_column_index_2,X_column_name_1='Null',X_column_name_2='Null',y_axis_min_max=[0.,0.]):
        """
            Plot the influence of the correlation between two variables in `self.X` on the errors made on the binary predictions self.y_pred (compared with self.y_true).

            Inputs:
                - X_column_index_1: Index of the first studied column in 'self.X'
                - X_column_index_2: Index of the second studied column in 'self.X'
                - X_column_name_1: Name of the variable represented in column X_column_index_1
                - X_column_name_2: Name of the variable represented in column X_column_index_2
                - y_axis_min_max: if defined, contains a list [y_axis_min,y_axis_max] with the min and max value on the y-axis in the plot
        """

        #observations of the variable of interest
        input_obs1=self.X[:,X_column_index_1].reshape(-1,1)
        input_obs1_mean=input_obs1.mean()
        input_obs1_std=input_obs1.std()
        input_obs1_scaled=(input_obs1-input_obs1_mean)/input_obs1_std

        input_obs2=self.X[:,X_column_index_2].reshape(-1,1)
        input_obs2_mean=input_obs2.mean()
        input_obs2_std=input_obs2.std()
        input_obs2_scaled=(input_obs2-input_obs2_mean)/input_obs2_std

        input_obs_scaled=np.concatenate((input_obs1_scaled,input_obs2_scaled),axis=1)

        corr_actual =  (input_obs1_scaled*input_obs2_scaled).sum()/(len(input_obs1_scaled)-1.)
        corr_plus=corr_actual+0.4
        if corr_plus>1.:
            corr_plus=1.
        corr_minus=corr_actual-0.4
        if corr_minus<-1.:
            corr_minus=-1.

        #extract the quantiles of interest
        list_corr=np.arange(corr_minus,corr_plus,0.07)

        #compute the impact of the stress
        list_res={'corr':[],'weight_means':[]}
        obs_strs=obs_stresser(input_obs_scaled)

        for i in range(len(list_corr)):
            #ksis=obs_strs.fit('covariance', [0, 1,list_corr[i]],gd_iterations=150)
            ksis=obs_strs.fit({'cov': [0, 1,list_corr[i]]},gd_iterations=150)
            lambdas=obs_strs.get_lambda()
            weight_mean=np.average(self.errors.flatten(), weights=lambdas)
            list_res['corr'].append(list_corr[i])
            list_res['weight_means'].append(weight_mean)

        plt.plot(list_res['corr'],list_res['weight_means'])
        if X_column_name_1=='Null' or X_column_name_2=='Null':
            plt.xlabel('Correlation')
        else:
            plt.xlabel('Correlation between '+X_column_name_1+' and '+X_column_name_2)
        plt.ylabel('Portion of errors')

        if y_axis_min_max[0]!=y_axis_min_max[1]:
            plt.ylim(ymax = y_axis_min_max[1], ymin = y_axis_min_max[0])

        plt.show()


    def show_variables_influence_on_pred(self, List_X_column_names=[],NonNegligibleOnly=False,delta_Q_stress=0.25):
        """
            Show the influence of all variables in `self.X` on the portion of predictions
            equal to 1, when their median is stressed toward their 1st and 3rd quantile.

            Inputs:
                - List_X_column_names: List containing the name of all variables (columns of `self.X`)
        """

        #init
        p=self.X.shape[1]

        if len(List_X_column_names)!=p:
            List_X_column_names=[]
            for i in range(p):
                List_X_column_names.append('var '+str(i))

        List_Stress_p=[]
        List_Stress_m=[]

        X_scaled=scale(self.X, with_mean=True, with_std=True, copy=True ,axis=0)

        obs_strs=obs_stresser(X_scaled)

        #compute the variables influence
        for i in range(p):
            list_StressValues_scaled=np.quantile(X_scaled[:,i], [0.25,0.5,0.75])
            #ksis_025=obs_strs.fit('mean', [i, list_StressValues_scaled[0]],gd_iterations=100)
            ksis_025=obs_strs.fit({'means': {i: list_StressValues_scaled[0]}},gd_iterations=100)
            lambdas_025=obs_strs.get_lambda()
            weight_mean_025=np.average(self.y_pred.flatten(), weights=lambdas_025)

            #ksis_050=obs_strs.fit('mean', [i, list_StressValues_scaled[1]],gd_iterations=100)
            ksis_050=obs_strs.fit({'means': {i: list_StressValues_scaled[1]}},gd_iterations=100)
            lambdas_050=obs_strs.get_lambda()
            weight_mean_050=np.average(self.y_pred.flatten(), weights=lambdas_050)

            #ksis_075=obs_strs.fit('mean', [i, list_StressValues_scaled[2]],gd_iterations=100)
            ksis_075=obs_strs.fit({'means': {i: list_StressValues_scaled[2]}},gd_iterations=100)
            lambdas_075=obs_strs.get_lambda()
            weight_mean_075=np.average(self.y_pred.flatten(), weights=lambdas_075)

            List_Stress_p.append(weight_mean_075-weight_mean_050)
            List_Stress_m.append(weight_mean_025-weight_mean_050)

            #

        #show meaningfull variables influence
        print('Negative Stress (S-) and Positive Stress (S+) on:')
        for i in range(p):
            if np.abs(List_Stress_p[i])>0.01 or np.abs(List_Stress_m[i])>0.01:
                print('S-: '+str("%.2f" % List_Stress_m[i])+'\tS+: '+str("%.2f" % List_Stress_p[i])+'\t <- '+List_X_column_names[i])

        #show other variables
        if NonNegligibleOnly==False:
            print("\nVariables with a negligible influence:")
            for i in range(p):
                if np.abs(List_Stress_p[i])<=0.01 and np.abs(List_Stress_m[i])<=0.01:
                    print('-> '+List_X_column_names[i])

    def show_covariance_influence_on_pred(self, List_X_column_names=[],NonNegligibleOnly=False):
        """
            NEW - TO  UPDATE
            Show the influence of covariance  variations between the pairs of variables in `self.X`.

            Inputs:
                - List_X_column_names: List containing the name of all variables (columns of `self.X`)

        """

        p=len(List_X_column_names)

        results=np.zeros([p*int(p/2)+1,4])

        nb_res=0
        for X_column_index_1 in range(p):
            for X_column_index_2 in range(p):
                if X_column_index_1<X_column_index_2:
                    #a) prepare the data
                    #observations of the variable of interest
                    input_obs1=self.X[:,X_column_index_1].reshape(-1,1)
                    input_obs1_mean=input_obs1.mean()
                    input_obs1_std=input_obs1.std()
                    input_obs1_scaled=(input_obs1-input_obs1_mean)/input_obs1_std

                    input_obs2=self.X[:,X_column_index_2].reshape(-1,1)
                    input_obs2_mean=input_obs2.mean()
                    input_obs2_std=input_obs2.std()
                    input_obs2_scaled=(input_obs2-input_obs2_mean)/input_obs2_std

                    input_obs_scaled=np.concatenate((input_obs1_scaled,input_obs2_scaled),axis=1)

                    #b) define the stress parameters
                    corr_actual =  (input_obs1_scaled*input_obs2_scaled).sum()/(len(input_obs1_scaled)-1.)
                    corr_plus=corr_actual+0.05
                    if corr_plus>1.:
                        corr_plus=1.
                    corr_minus=corr_actual-0.05
                    if corr_minus<-1.:
                        corr_minus=-1.

                    #extract the quantiles of interest
                    list_corr=np.array([corr_minus,corr_actual,corr_plus])

                    #c) compute the impact of the stress
                    list_res={'corr':[],'weight_means':[]}
                    obs_strs=obs_stresser(input_obs_scaled)

                    for i in range(len(list_corr)):
                        #ksis=obs_strs.fit('covariance', [0, 1,list_corr[i]],gd_iterations=150)
                        ksis=obs_strs.fit({'cov': [0, 1,list_corr[i]]},gd_iterations=150)
                        lambdas=obs_strs.get_lambda()
                        weight_mean=np.average(self.y_pred.flatten(), weights=lambdas)
                        list_res['corr'].append(list_corr[i])
                        list_res['weight_means'].append(weight_mean)

                    #d) show the results
                    score=np.abs(list_res['weight_means'][2]-list_res['weight_means'][1])+np.abs(list_res['weight_means'][1]-list_res['weight_means'][0])
                    results[nb_res,0]=score
                    results[nb_res,1]=list_res['corr'][1]
                    results[nb_res,2]=X_column_index_1
                    results[nb_res,3]=X_column_index_2
                    nb_res+=1

        #rank the results
        rnk=np.argsort(-results[:,0])
        for i in range(nb_res):
            obs_id=rnk[i]
            print(List_X_column_names[results[obs_id,2]]+' and '+List_X_column_names[results[obs_id,3]]+': ',
                np.round(results[obs_id,0],4),
                np.round(results[obs_id,1],4))

        return results



    def find_observation_impacted_by_sensitive_variable(self, X_column_index_S):
        """
            Find the observations whose predictions are the most impacted by a sensitive variable

            Inputs:
                - X_column_index_S: Index of the column in 'self.X'
        """

        #observations of the variable of interest
        input_obs_S=self.X[:,X_column_index_S].reshape(-1,1)
        input_obs_S_mean=input_obs_S.mean()
        input_obs_S_std=input_obs_S.std()
        input_obs_S_scaled=(input_obs_S-input_obs_S_mean)/input_obs_S_std

        input_obs_yp=self.y_pred.reshape(-1,1)
        input_obs_yp_mean=input_obs_yp.mean()
        input_obs_yp_std=input_obs_yp.std()
        input_obs_yp_scaled=(input_obs_yp-input_obs_yp_mean)/input_obs_yp_std

        input_obs_scaled=np.concatenate((input_obs_S_scaled,input_obs_yp_scaled),axis=1)

        corr_actual =  (input_obs_S_scaled*input_obs_yp_scaled).sum()/(len(input_obs_S_scaled)-1.)

        corr_plus=corr_actual+0.05
        if corr_plus>1.:
            corr_plus=1.
        corr_minus=corr_actual-0.05
        if corr_minus<-1.:
            corr_minus=-1.

        obs_strs=obs_stresser(input_obs_scaled)

        #ksis_plus=obs_strs.fit('covariance', [0, 1,corr_plus],gd_iterations=150)
        ksis_plus=obs_strs.fit({'cov': [0, 1,corr_plus]},gd_iterations=150)
        lambdas_plus=obs_strs.get_lambda()

        #ksis_minus=obs_strs.fit('covariance', [0, 1,corr_minus],gd_iterations=150)
        ksis_minus=obs_strs.fit({'cov': [0, 1,corr_minus]},gd_iterations=150)
        lambdas_minus=obs_strs.get_lambda()

        lambda_delta=lambdas_plus-lambdas_minus

        Xcr=self.X-self.X.mean(axis=0)
        Xcr=Xcr/Xcr.std(axis=0)

        influence_variables=np.average(Xcr, weights=lambda_delta,axis=0)

        return lambda_delta , influence_variables



def compare_mean_influence_on_pred(explainer1,explainer2, X_column_index,X_column_name='Null',explainer1_name='Null',explainer2_name='Null',y_axis_min_max=[0.,0.]):
        """
            Compare, in two explainers, the influence of their variable mean in `self.X`  on the binary predictions self.y_pred.

            Inputs:
                - explainer1,explainer2: the two explainers
                - X_column_index: Index of the column that will be studied in the explainer's X
                - X_column_name: Name of the treated variable in X
                - explainer1_name,explainer2_name: name of the two explainers
                - y_axis_min_max: if defined, contains a list [y_axis_min,y_axis_max] with the min and max value on the y-axis in the plot
        """

        #influence on all data
        quantiles1,scores1=explainer1.plot_mean_influence_on_pred(X_column_index,X_column_name=X_column_name,y_axis_min_max=[y_axis_min_max[0],y_axis_min_max[1]],plot_results=False)
        quantiles2,scores2=explainer2.plot_mean_influence_on_pred(X_column_index,X_column_name=X_column_name,y_axis_min_max=[y_axis_min_max[0],y_axis_min_max[1]],plot_results=False)

        #variability of the scores...
        #... quantiles of the variable values in both explainers
        Studied_X1=explainer1.Get_X()
        Studied_y_pred1=explainer1.Get_y_pred()

        Studied_X2=explainer2.Get_X()
        Studied_y_pred2=explainer2.Get_y_pred()

        input_obs1=Studied_X1[:,X_column_index].reshape(-1,1)
        input_obs_scaled1=Studied_X1[:,X_column_index].reshape(-1,1)
        input_obs_scaled1 = scale(input_obs_scaled1, with_mean=True, with_std=True, copy=True )

        q1=np.quantile(input_obs1, 0.1)
        q9=np.quantile(input_obs1, 0.9)
        nq1=np.quantile(input_obs_scaled1, 0.1)
        nq9=np.quantile(input_obs_scaled1, 0.9)
        list_StressValues_1=np.arange(q1,q9,(q9-q1)/5.001)
        list_StressValues_scaled_1=np.arange(nq1,nq9,(nq9-nq1)/5.001)

        input_obs2=Studied_X2[:,X_column_index].reshape(-1,1)
        input_obs_scaled2=Studied_X2[:,X_column_index].reshape(-1,1)
        input_obs_scaled2 = scale(input_obs_scaled2, with_mean=True, with_std=True, copy=True )
        q1=np.quantile(input_obs2, 0.1)
        q9=np.quantile(input_obs2, 0.9)
        nq1=np.quantile(input_obs_scaled2, 0.1)
        nq9=np.quantile(input_obs_scaled2, 0.9)
        list_StressValues_2=np.arange(q1,q9,(q9-q1)/5.001)
        list_StressValues_scaled_2=np.arange(nq1,nq9,(nq9-nq1)/5.001)

        #stress the samples and compute the average positive predictions of each stressed sample
        NbRandomSamples=80

        ScoreQuantile_1_1=np.zeros(len(list_StressValues_1))
        ScoreQuantile_1_9=np.zeros(len(list_StressValues_1))

        ScoreQuantile_2_1=np.zeros(len(list_StressValues_1))
        ScoreQuantile_2_9=np.zeros(len(list_StressValues_1))

        osm1=obs_stresser_multiple(input_obs_scaled1,NbRandomSamples,int(input_obs_scaled1.shape[0]*0.5))
        osm2=obs_stresser_multiple(input_obs_scaled2,NbRandomSamples,int(input_obs_scaled2.shape[0]*0.5))

        for i in range(len(list_StressValues_scaled_1)):
            #osm1.fit_and_cpt_lambdas('mean', [0, list_StressValues_scaled_1[i]],gd_iterations=150)
            osm1.fit_and_cpt_lambdas({'means': {0: list_StressValues_scaled_1[i]}},gd_iterations=150)
            LstQuantiles=osm1.CptQuantilesOfWeightedSumsWithLambdas(Studied_y_pred1,[0.1,0.9])
            ScoreQuantile_1_1[i]=LstQuantiles[0]
            ScoreQuantile_1_9[i]=LstQuantiles[1]

        for i in range(len(list_StressValues_scaled_2)):
            #osm2.fit_and_cpt_lambdas('mean', [0, list_StressValues_scaled_2[i]],gd_iterations=150)
            osm2.fit_and_cpt_lambdas({'means': {0: list_StressValues_scaled_2[i]}},gd_iterations=150)
            LstQuantiles=osm2.CptQuantilesOfWeightedSumsWithLambdas(Studied_y_pred2,[0.1,0.9])
            ScoreQuantile_2_1[i]=LstQuantiles[0]
            ScoreQuantile_2_9[i]=LstQuantiles[1]

        #show the results

        if explainer1_name=='Null':
            explainer1_name='explainer 1'

        if explainer2_name=='Null':
            explainer2_name='explainer 2'

        plt.plot(quantiles1,scores1,'r', label=explainer1_name)
        plt.fill_between(list_StressValues_1, ScoreQuantile_1_1, ScoreQuantile_1_9, alpha=0.2,color='r')
        plt.plot(quantiles2,scores2,'b', label=explainer2_name)
        plt.fill_between(list_StressValues_2, ScoreQuantile_2_1, ScoreQuantile_2_9, alpha=0.2,color='b')
        plt.legend(fontsize=10)

        if y_axis_min_max[0]!=y_axis_min_max[1]:
            plt.ylim(ymax = y_axis_min_max[1], ymin = y_axis_min_max[0])

        if X_column_name=='Null':
            plt.xlabel('Mean value')
        else:
            plt.xlabel('Mean of '+X_column_name)
        plt.ylabel('Portion predicted 1s')

        plt.show()
