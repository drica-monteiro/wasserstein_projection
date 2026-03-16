import numpy as np
import pandas as pd
import sklearn as sk
import os
from numpy import random
from tqdm import tqdm

#+++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++
def GenerateBiasedData(n_F,n_M):
    """
    GenerateData(n_F,n_M) -> n_F,n_M are the number of observations for females and males

    Properties of the simulated data:
     - 4 quantitative data (0,1,2,3) and 3 binary data (4,5,6)
     - All quantitative data are correlated
     - All binary data are related
     - For females
       - Binary variables (4,5) have a strong impact on variable 0
       - Binary variable 6 has a strong impact on variable 1
       - Means (9,8,10,10) for variables (0,1,2,3)
     - For Males
       - Binary variables (4,5) have little impact on variable 0
       - Binary variable 6 has no impact on variable 1
       - Means (10,10,10,10) for variables (0,1,2,3)
     - The link between the X and Y is also slighly different for males and females
    """

    #generate female data
    covF=np.array(
        [[0.2 , 0.5  , 0.5 , 0.5 , 2.  , 2.  , 0  ],
         [0.5  , 0.2  , 0.5 , 0.5 , 0.  , 0.  , 4.],
         [0.5  , 0.5  , 0.2 , 0.5 , 0.  , 0.  , 0  ],
         [0.5  , 0.5  , 0.5 , 0.2 , 0.  , 0.  , 0  ],
         [2.   , 0    , 0   , 0.  , 0.2 , 0.9 , 0.9],
         [2.   , 0    , 0   , 0   , 0.9 , 0.2 , 0.9],
         [0.   , 4.0  , 0   , 0   , 0.9 , 0.9 , 0.2]]
    )

    meanF = np.array([9.,8.,10.,10.,0.,0.,0.])

    X_F=np.round(random.multivariate_normal(meanF, covF, size=n_F),1)

    toto_m=X_F[:,:]<0.
    toto_m[:,:4]=False
    toto_p=X_F[:,:]>0.
    toto_p[:,:4]=False
    X_F[toto_m]=0.
    X_F[toto_p]=1.

    thetaF=np.array([1.,1.0,-1.0,1.0,0.3,0.3,0.3])
    Y_F = np.round(np.dot(X_F,thetaF.reshape(-1,1)) + 0.2*np.random.normal(size=n_F).reshape(-1,1),1)



    #generate male data
    covM=np.array(
        [[0.2 , 0.5  , 0.5 , 0.5 , 0.5 , 0.5 , 0  ],
         [0.5  , 0.2  , 0.5 , 0.5 , 0.  , 0.  , 0.],
         [0.5  , 0.5  , 0.2 , 0.5 , 0.  , 0.  , 0  ],
         [0.5  , 0.5  , 0.5 , 0.2 , 0.  , 0.  , 0  ],
         [0.5  , 0    , 0   , 0.  , 0.2 , 0.9 , 0.9],
         [0.5  , 0    , 0   , 0   , 0.9 , 0.2 , 0.9],
         [0.   , 0.0  , 0   , 0   , 0.9 , 0.9 , 0.2]]
    )

    meanM = np.array([10.,10.,10.,10.,0.,0.,0.])

    X_M=np.round(random.multivariate_normal(meanM, covM, size=n_M),1)

    toto_m=X_M[:,:]<0.
    toto_m[:,:4]=False
    toto_p=X_M[:,:]>0.
    toto_p[:,:4]=False
    X_M[toto_m]=0.
    X_M[toto_p]=1.

    thetaM=np.array([1.,0.5,-0.5,1.0,0.3,0.3,0.3])
    Y_M = np.round(np.dot(X_M,thetaM.reshape(-1,1)) + 0.2*np.random.normal(size=n_M).reshape(-1,1),1)

    #merge data
    X=np.concatenate((X_F,X_M),axis=0)
    Y=np.concatenate((Y_F,Y_M),axis=0)
    S=np.zeros(n_F+n_M)
    S[n_F:]=1

    #shuffle data
    toto=np.arange(n_F+n_M)
    np.random.shuffle(toto)
    X=X[toto,:]
    Y=Y[toto,:]
    S=S[toto]

    return X,Y,S


#+++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++

def GenerateData2(n_F,n_M):
    """
    Same as GenerateData with other parameters
    """

    #generate female data
    covF=np.array(
        [[1.0 , 0.5  , 0.5 , 0.5  , 1.  , 1.  , 0  ],
         [0.5  , 1.0  , 0.5 , 0.5 , 0.  , 0.  , 4.],
         [0.5  , 0.5  , 1.0 , 0.5 , 0.  , 0.  , 0  ],
         [0.5  , 0.5  , 0.5 , 1.0 , 0.  , 0.  , 0  ],
         [1.   , 0    , 0   , 0.  , 1.0 , 0.2 , 0.2],
         [1.   , 0    , 0   , 0   , 0.2 , 1.0 , 0.9],
         [0.   , 4.0  , 0   , 0   , 0.2 , 0.2 , 1.0]]
    )

    meanF = np.array([10.,6.,10.,8.,0.,0.,0.])

    X_F=np.round(random.multivariate_normal(meanF, covF, size=n_F),1)

    toto_m=X_F[:,:]<0.
    toto_m[:,:4]=False
    toto_p=X_F[:,:]>0.
    toto_p[:,:4]=False
    X_F[toto_m]=0.
    X_F[toto_p]=1.

    #thetaF=np.array([1.5,1.5,0.5,0.5,0.0,0.0,0.0])
    thetaF=np.array([1.,1.,1.,1.,0.0,0.0,0.0])
    Y_F = np.round(np.dot(X_F,thetaF.reshape(-1,1)) + 0.1*np.random.normal(size=n_F).reshape(-1,1),1)



    #generate male data
    covM=np.array(
        [[1.0 , 0.5  , 0.5 , 0.5 ,  1.0 , 1.0 , 0  ],
         [0.5  , 1.0  , 0.5 , 0.5 , 0.  , 0.  , 0.],
         [0.5  , 0.5  , 1.0 , 0.5 , 0.  , 0.  , 0  ],
         [0.5  , 0.5  , 0.5 , 1.0 , 0.  , 0.  , 0  ],
         [1.0  , 0    , 0   , 0.  , 1.0 , 0.2 , 0.2],
         [1.0  , 0    , 0   , 0   , 0.2 , 1.0 , 0.2],
         [0.   , 0.0  , 0   , 0   , 0.2 , 0.2 , 1.0]]
    )

    meanM = np.array([10.,10.,8.,10.,0.,0.,0.])

    X_M=np.round(random.multivariate_normal(meanM, covM, size=n_M),1)

    toto_m=X_M[:,:]<0.
    toto_m[:,:4]=False
    toto_p=X_M[:,:]>0.
    toto_p[:,:4]=False
    X_M[toto_m]=0.
    X_M[toto_p]=1.

    thetaM=np.array([1.,1.,1.,1.0,0.0,0.0,0.0])
    Y_M = np.round(np.dot(X_M,thetaM.reshape(-1,1)) + 0.1*np.random.normal(size=n_M).reshape(-1,1),1)

    #merge data
    X=np.concatenate((X_F,X_M),axis=0)
    Y=np.concatenate((Y_F,Y_M),axis=0)
    S=np.zeros(n_F+n_M)
    S[n_F:]=1

    #shuffle data
    toto=np.arange(n_F+n_M)
    np.random.shuffle(toto)
    X=X[toto,:]
    Y=Y[toto,:]
    S=S[toto]

    return X,Y,S



#+++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++ +++

def Generate_HR_Data(n):
    """
    Generate a table representing HR data.
    -> n profiles are generated
    -> The parameters of the generated data are hard coded in the function
    -> The generated table contains categorical and continuous variables. It is returned as a pandas dataframe
    """

    #+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
    #1) functions and classes to help generating the data
    #+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
    def Normalize_dic_weights(treated_dic):
        sum_weights=0.
        for key in treated_dic.keys():
            sum_weights+=treated_dic[key]

        for key in treated_dic.keys():
            treated_dic[key]/=sum_weights

    #1.1) jobs
    class Job:
        """
        Class that contains the parameters of a job
        """

        def __init__(self,Name,PortionF,baseline_salary,annual_evo_salary,delta_salary_w_kids_M,
                     delta_salary_w_kids_F,delta_salary_wo_kids_M,delta_salary_wo_kids_F,
                     distrib_years_in_position,distrib_years_studies,distrib_performance,
                     distrib_hours_per_week):
            """
            (string) Na'ME': job name
            (float) PortionF: Portion of females doing this job. Value in [0,1]
            (float) baseline_salary: Standard baseline annual salary for this job
            (float) annual_evo_salary: Annual standard evolution of the annual salary for this job
            (float) delta_salary_w_kids_M: Bonus/malus on the annual salary for males with kids
            (float) delta_salary_w_kids_F: Bonus/malus on the annual salary for females with kids
            (float) delta_salary_wo_kids_M: Bonus/malus on the annual salary for males without kids
            (float) delta_salary_wo_kids_F: Bonus/malus on the annual salary for females without kids
            (dict) distrib_years_in_position: dict countaining the distribution of the possible years
                                              in this position
            (dict) distrib_years_studies: dict countaining the distribution of the years of study for
                                          this position
            (dict) distrib_performance:  dict countaining the distribution of the performance in this
                                         position. The Huntington Ingalls system is used:
                                         5 = FE (Far Exceeds) / 4 = EX (Exceeds Expectations) /
                                         3 = ME (Meets Expectations) / 2 = DR (Development Required) /
                                         1 = IR (Improvement Required)
            (dict) distrib_hours_per_week: distribution of the hours worked per week
            """
            #self.Name
            self.Name=Name

            #self.PortionF
            self.PortionF=PortionF
            if self.PortionF>1.:
                self.PortionF=1
            if self.PortionF<0.:
                self.PortionF=0

            #baseline_salary
            self.baseline_salary=baseline_salary

            #annual_evo_salary
            self.annual_evo_salary=annual_evo_salary

            #delta_salary_w_kids_M/delta_salary_w_kids_F/delta_salary_wo_kids_M/delta_salary_wo_kids_F
            self.delta_salary_w_kids_M=delta_salary_w_kids_M
            self.delta_salary_w_kids_F=delta_salary_w_kids_F
            self.delta_salary_wo_kids_M=delta_salary_wo_kids_M
            self.delta_salary_wo_kids_F=delta_salary_wo_kids_F

            #self.years_in_position_keys and self.years_in_position_proba
            Normalize_dic_weights(distrib_years_in_position)
            self.years_in_position_keys=list(distrib_years_in_position.keys())
            self.years_in_position_proba=[]
            for key in self.years_in_position_keys:
                self.years_in_position_proba.append(distrib_years_in_position[key])

            #self.years_studies_keys and self.years_studies_proba
            Normalize_dic_weights(distrib_years_studies)
            self.years_studies_keys=list(distrib_years_studies.keys())
            self.years_studies_proba=[]
            for key in self.years_studies_keys:
                self.years_studies_proba.append(distrib_years_studies[key])

            #self.performance_keys and self.performance_proba
            Normalize_dic_weights(distrib_performance)
            self.performance_keys=list(distrib_performance.keys())
            self.performance_proba=[]
            for key in self.performance_keys:
                self.performance_proba.append(distrib_performance[key])

            #self.hours_per_week_keys and self.hours_per_week_proba
            Normalize_dic_weights(distrib_hours_per_week)
            self.hours_per_week_keys=list(distrib_hours_per_week.keys())
            self.hours_per_week_proba=[]
            for key in self.hours_per_week_keys:
                self.hours_per_week_proba.append(distrib_hours_per_week[key])

        def draw_gender(self):
            if random.rand()<self.PortionF:
                return 'F'
            else:
                return 'M'

        def draw_years_in_position(self):
            return random.choice(self.years_in_position_keys,p=self.years_in_position_proba)

        def draw_years_studies(self):
            return random.choice(self.years_studies_keys,p=self.years_studies_proba)

        def draw_performance(self):
            return random.choice(self.performance_keys,p=self.performance_proba)

        def draw_hours_per_week(self):
            return random.choice(self.hours_per_week_keys,p=self.hours_per_week_proba)

    #1.2) job locations
    class JobLocation:
        """
        Class that contains the parameters of a job location
        """

        def __init__(self,Name,Importance,Delta_salary_M,Delta_salary_F,PortionJobs):
            """
            (string) Na'ME': Name of the job location
            (float) Importance: Importance of this location. The job location of the
                                profiles is drawn according to this value
            (float) Delta_salary_M: bonus or malus of the salary for Males when working
                                    at this location
            (float) Delta_salary_M: bonus or malus of the salary for Females when working
                                    at this location
            (dict) DistribJobs: Distribution of the jobs at this location. One entry is
                                "[JobName]: weight", where the sum of all weight should
                                be 1 (they will be normalized otherwise).
            """
            #self.Name
            self.Name=Name

            #self.Importance
            self.Importance=Importance

            #self.Delta_salary_M
            self.Delta_salary_M=Delta_salary_M

            #self.Delta_salary_F
            self.Delta_salary_F=Delta_salary_F

            #self.Jobs_keys and self.Jobs_proba
            Normalize_dic_weights(PortionJobs)
            self.Jobs_keys=list(PortionJobs.keys())
            self.Jobs_proba=[]
            for key in self.Jobs_keys:
                self.Jobs_proba.append(PortionJobs[key])

        def drawJob(self):
            return random.choice(self.Jobs_keys,p=self.Jobs_proba)



    #+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
    #2) define the job and job location properties
    #+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + +

    #2.1) jobs
    PortionF=0.7
    baseline_salary=30000.
    annual_evo_salary=2000.
    delta_salary_w_kids_M=0.
    delta_salary_w_kids_F=-2000.
    delta_salary_wo_kids_M=0.
    delta_salary_wo_kids_F=-500.
    distrib_years_in_position={0:10,1:7,2:7,3:6,4:6,5:5,6:5,7:5,8:5,9:4,10:4}
    distrib_years_studies={3:10,5:20,6:20,8:4}
    distrib_performance={'IR':3,'DR':11,'ME':42,'EX':15,'FE':4}  #from Improvement Required to Far Exceeds
    distrib_hours_per_week={35:10,40:20}
    J1=Job('AdministrativeAssistant',PortionF,baseline_salary,annual_evo_salary,delta_salary_w_kids_M,
                 delta_salary_w_kids_F,delta_salary_wo_kids_M,delta_salary_wo_kids_F,
                 distrib_years_in_position,distrib_years_studies,distrib_performance,
                 distrib_hours_per_week)

    PortionF=0.56
    baseline_salary=45000.
    annual_evo_salary=4000.
    delta_salary_w_kids_M=0.
    delta_salary_w_kids_F=-4000.
    delta_salary_wo_kids_M=0.
    delta_salary_wo_kids_F=-1000.
    distrib_years_in_position={0:10,1:6,2:5,3:4,4:3,5:2,6:1,7:1,8:1,9:1,10:1}
    distrib_years_studies={6:10,7:2,8:20,11:3}  #baseline is French brevet des colleges
    distrib_performance={'IR':3,'DR':11,'ME':42,'EX':15,'FE':4}  #from Improvement Required to Far Exceeds
    distrib_hours_per_week={35:6,40:54}
    J2=Job('BusinessAnalyst',PortionF,baseline_salary,annual_evo_salary,delta_salary_w_kids_M,
                 delta_salary_w_kids_F,delta_salary_wo_kids_M,delta_salary_wo_kids_F,
                 distrib_years_in_position,distrib_years_studies,distrib_performance,
                 distrib_hours_per_week)

    PortionF=0.40
    baseline_salary=120000.
    annual_evo_salary=8000.
    delta_salary_w_kids_M=0.
    delta_salary_w_kids_F=-20000.
    delta_salary_wo_kids_M=0.
    delta_salary_wo_kids_F=-5000.
    distrib_years_in_position={0:10,1:6,2:3}
    distrib_years_studies={8:20,11:4}  #baseline is French brevet des colleges
    distrib_performance={'IR':1,'DR':2,'ME':10,'EX':10,'FE':2}  #from Improvement Required to Far Exceeds
    distrib_hours_per_week={35:1,40:50}
    J3=Job('ConsultantExpert',PortionF,baseline_salary,annual_evo_salary,delta_salary_w_kids_M,
                 delta_salary_w_kids_F,delta_salary_wo_kids_M,delta_salary_wo_kids_F,
                 distrib_years_in_position,distrib_years_studies,distrib_performance,
                 distrib_hours_per_week)

    PortionF=0.48
    baseline_salary=80000.
    annual_evo_salary=5000.
    delta_salary_w_kids_M=0.
    delta_salary_w_kids_F=-5000.
    delta_salary_wo_kids_M=0.
    delta_salary_wo_kids_F=-1250.
    distrib_years_in_position={0:10,1:6,2:3}
    distrib_years_studies={8:20,11:1}  #baseline is French brevet des colleges
    distrib_performance={'IR':3,'DR':4,'ME':10,'EX':4,'FE':1}  #from Improvement Required to Far Exceeds
    distrib_hours_per_week={35:5,40:50}
    J4=Job('Consultant',PortionF,baseline_salary,annual_evo_salary,delta_salary_w_kids_M,
                 delta_salary_w_kids_F,delta_salary_wo_kids_M,delta_salary_wo_kids_F,
                 distrib_years_in_position,distrib_years_studies,distrib_performance,
                 distrib_hours_per_week)

    PortionF=0.52
    baseline_salary=60000.
    annual_evo_salary=4000.
    delta_salary_w_kids_M=0.
    delta_salary_w_kids_F=-4000.
    delta_salary_wo_kids_M=0.
    delta_salary_wo_kids_F=-1000.
    distrib_years_in_position={0:10,1:6,2:5,3:4,4:3,5:2,6:1,7:1,8:1,9:1,10:1}
    distrib_years_studies={6:10,8:15}  #baseline is French brevet des colleges
    distrib_performance={'IR':1,'DR':2,'ME':10,'EX':4,'FE':1}  #from Improvement Required to Far Exceeds
    distrib_hours_per_week={35:8,40:50}
    J5=Job('ProjectManager',PortionF,baseline_salary,annual_evo_salary,delta_salary_w_kids_M,
                 delta_salary_w_kids_F,delta_salary_wo_kids_M,delta_salary_wo_kids_F,
                 distrib_years_in_position,distrib_years_studies,distrib_performance,
                 distrib_hours_per_week)

    PortionF=0.5
    baseline_salary=30000.
    annual_evo_salary=2000.
    delta_salary_w_kids_M=0.
    delta_salary_w_kids_F=-2000.
    delta_salary_wo_kids_M=0.
    delta_salary_wo_kids_F=-500.
    distrib_years_in_position={0:10,1:7,2:7,3:6,4:6,5:5,6:5,7:5,8:5,9:4,10:4}
    distrib_years_studies={3:20,5:20,6:2,8:2}
    distrib_performance={'IR':3,'DR':11,'ME':42,'EX':15,'FE':1}  #from Improvement Required to Far Exceeds
    distrib_hours_per_week={35:20,40:20}
    J6=Job('ServiceOperator',PortionF,baseline_salary,annual_evo_salary,delta_salary_w_kids_M,
                 delta_salary_w_kids_F,delta_salary_wo_kids_M,delta_salary_wo_kids_F,
                 distrib_years_in_position,distrib_years_studies,distrib_performance,
                 distrib_hours_per_week)

    PortionF=0.3
    baseline_salary=52000.
    annual_evo_salary=6000.
    delta_salary_w_kids_M=0.
    delta_salary_w_kids_F=-6000.
    delta_salary_wo_kids_M=0.
    delta_salary_wo_kids_F=-1500.
    distrib_years_in_position={0:10,1:6,2:5,3:4,4:3,5:2,6:1,7:1,8:1,9:1,10:1}
    distrib_years_studies={5:20,6:20,8:40,11:5}
    distrib_performance={'IR':1,'DR':11,'ME':42,'EX':11,'FE':1}  #from Improvement Required to Far Exceeds
    distrib_hours_per_week={35:10,40:20}
    J7=Job('IT',PortionF,baseline_salary,annual_evo_salary,delta_salary_w_kids_M,
                 delta_salary_w_kids_F,delta_salary_wo_kids_M,delta_salary_wo_kids_F,
                 distrib_years_in_position,distrib_years_studies,distrib_performance,
                 distrib_hours_per_week)

    Jobs={'AdministrativeAssistant':J1,'BusinessAnalyst':J2,'ConsultantExpert':J3,'Consultant':J4,'ProjectManager':J5,'ServiceOperator':J6,'IT':J7}

    #2.2) job locations
    Importance=10
    Delta_salary_M=0.
    Delta_salary_F=0.
    PortionJobs={'AdministrativeAssistant':10,'BusinessAnalyst':2,'ConsultantExpert':10,'Consultant':40,'ProjectManager':10,'ServiceOperator':20,'IT':5}
    JL1=JobLocation('Loc1',Importance,Delta_salary_M,Delta_salary_F,PortionJobs)

    Importance=3
    Delta_salary_M=0.
    Delta_salary_F=0.
    PortionJobs={'AdministrativeAssistant':3,'BusinessAnalyst':0,'ConsultantExpert':2,'Consultant':2,'ProjectManager':2,'ServiceOperator':5,'IT':20}
    JL2=JobLocation('Loc2',Importance,Delta_salary_M,Delta_salary_F,PortionJobs)

    Importance=1
    Delta_salary_M=0.
    Delta_salary_F=0.
    PortionJobs={'AdministrativeAssistant':5,'BusinessAnalyst':15,'ConsultantExpert':2,'Consultant':2,'ProjectManager':2,'ServiceOperator':3,'IT':5}
    JL3=JobLocation('Loc2',Importance,Delta_salary_M,Delta_salary_F,PortionJobs)

    Importance=1
    Delta_salary_M=0.
    Delta_salary_F=0.
    PortionJobs={'AdministrativeAssistant':8,'BusinessAnalyst':0,'ConsultantExpert':0,'Consultant':0,'ProjectManager':8,'ServiceOperator':40,'IT':2}
    JL4=JobLocation('Loc2',Importance,Delta_salary_M,Delta_salary_F,PortionJobs)

    JobLocations={'Loc1':JL1,'Loc2':JL2,'Loc3':JL3,'Loc4':JL4}

    #+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
    #3) draw the profiles
    #+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + +

    JL_keys=list(JobLocations.keys())
    JL_proba=[]
    loc_sum=0.
    for key in JL_keys:
        loc_sum+=JobLocations[key].Importance
    for key in JL_keys:
        JL_proba.append(JobLocations[key].Importance/loc_sum)

    EmployeeProfiles = pd.DataFrame(columns = ["location","occupation","gender","years_in_position","years_studies","performance","hours_per_week","kids","annual_salary","annual_bonus","satisfactory"])

    print('Generate data')
    for i in tqdm(range(n)):
        #3.1) draw the job location
        JobLocName=random.choice(JL_keys,p=JL_proba)

        #3.2) draw the job and draw/get other personal features
        JobName=JobLocations[JobLocName].drawJob()

        gender=Jobs[JobName].draw_gender()
        years_in_position=Jobs[JobName].draw_years_in_position()
        years_studies=Jobs[JobName].draw_years_studies()
        performance=Jobs[JobName].draw_performance()
        hours_per_week=Jobs[JobName].draw_hours_per_week()
        baseline_salary=Jobs[JobName].baseline_salary
        annual_evo_salary=Jobs[JobName].annual_evo_salary


        if years_in_position<2:
            KidsNoKids=random.choice(['Kids','NoKids'],p=[0.2,0.8])
        else:
            KidsNoKids=random.choice(['Kids','NoKids'],p=[0.8,0.2])

        #3.3) get profile related parameters
        if gender=='F':
            JL_DeltaSalary=JobLocations[JobLocName].Delta_salary_F
            if KidsNoKids=='Kids':   #FEMALE WITH KIDS    ('IR','DR','ME','EX','FE')
                J_DeltaSalary=Jobs[JobName].delta_salary_w_kids_F
                if performance=='DR':
                    performance=random.choice(['IR','DR'],p=[0.2,0.8])
                if performance=='ME':
                    performance=random.choice(['DR','ME'],p=[0.3,0.7])
                if performance=='EX':
                    performance=random.choice(['ME','EX'],p=[0.5,0.5])
                if performance=='FE':
                    performance=random.choice(['EX','FE'],p=[0.7,0.3])
            else:   #FEMALE WITHOUT KIDS
                J_DeltaSalary=Jobs[JobName].delta_salary_wo_kids_F
        else:
            JL_DeltaSalary=JobLocations[JobLocName].Delta_salary_M
            if KidsNoKids=='Kids':   #MALE WITH KIDS
                J_DeltaSalary=Jobs[JobName].delta_salary_w_kids_M
                if performance=='DR':
                    performance=random.choice(['IR','DR'],p=[0.1,0.9])
                if performance=='ME':
                    performance=random.choice(['DR','ME'],p=[0.1,0.9])
                if performance=='EX':
                    performance=random.choice(['ME','EX'],p=[0.1,0.9])
                if performance=='FE':
                    performance=random.choice(['EX','FE'],p=[0.1,0.9])
            else:   #MALE WITHOUT KIDS
                J_DeltaSalary=Jobs[JobName].delta_salary_wo_kids_M

        #3.4) compute the annual salary and the annual bonus

        #... coefs
        if performance=='IR':
            coef_performance=0.1
        elif performance=='DR':
            coef_performance=0.5
        elif performance=='ME':
            coef_performance=1.0
        elif performance=='EX':
            coef_performance=1.2
        elif performance=='FE':
            coef_performance=1.5

        coef_hours_per_week=hours_per_week/40.

        #... annal salary
        annual_salary=int((baseline_salary+annual_evo_salary*years_in_position*coef_performance+JL_DeltaSalary+J_DeltaSalary)*coef_hours_per_week + baseline_salary*random.randn()/30.)

        baseline_annual_salary=(baseline_salary+annual_evo_salary*years_in_position)*coef_hours_per_week

        #... annual bonus
        score_for_bonus=0.7*coef_performance+0.2*(baseline_salary/100000)+0.1*random.rand()

        if score_for_bonus>1.:
            annual_bonus="Yes"
        else:
            annual_bonus="No"

        #... satisfactory level
        satisfactory="Yes"
        if (annual_salary<baseline_annual_salary*0.95):
            satisfactory="No"
        if satisfactory=="No" and annual_bonus=="Yes":
            satisfactory=random.choice(['Yes','No'],p=[0.8,0.2])

        #3.5) add the profile
        new_employee={"location":JobLocName,
                      "occupation":JobName,
                      "gender":gender,
                      "years_in_position":years_in_position,
                      "years_studies":years_studies,
                      "performance":performance,
                      "hours_per_week":hours_per_week,
                      "kids":KidsNoKids,
                      "annual_salary":annual_salary,
                      "annual_bonus":annual_bonus,
                      "satisfactory":satisfactory
                      }

        EmployeeProfiles.loc[len(EmployeeProfiles)]=new_employee

    return EmployeeProfiles
