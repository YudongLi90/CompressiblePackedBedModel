import numpy as np
from ClassicKCAnalyzer import ClassicKCAnalyzer
from ColumnExperiment import ColumnExperiment
from ColumnPlots import ColumnPlots
from Density import Density
from Porosity import Porosity
from ModifiedKCAnalyzer import ModifiedKCAnalyzer

from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy.optimize import leastsq
from scipy.integrate import solve_ivp
from scipy.integrate import odeint

class PackedBed:
    g = 9.8
    rouf = 1000
    solids = 0.2
    def __init__(self, exp:ColumnExperiment,density:Density=None,porosity:Porosity=None,frictionCoeff=0.1,scaleUp=False,solids=0.17):
        self.name = exp.name
        self.exp = exp
        if isinstance(density,Density): # recommend use cleaned data (exp) (manual cleaned more prefered)
            self.density = density
        else: # you might want to use density before data clean up (but need to be careful to check whether density have outliers)
            self.density = Density(exp)
        if isinstance(porosity,Porosity):
            self.porosity = porosity
        else:
            self.porosity = Porosity(exp)
        self.modifiedKC = ModifiedKCAnalyzer(exp,self.porosity)
        self.miu = frictionCoeff
        self.radius = exp.radius # m
        self.area = exp.area # m2
        self.scaleUp = scaleUp
        self.viscosity = exp.viscosity
        result_fitting = self.exp.result
        self.bedHeight = [result_fitting[i]['bedHeight'] for i in range(len(result_fitting))]
        self.packingDensity = [result_fitting [i]['packingDensity'] for i in range(len(result_fitting))] # unit: kg/
        #self.compute_Ps_related()
        


    def getBedProperties(self,frictionCoeff,U,BedHeight,radius,yinit,DensityCoeff=np.zeros(1),\
                            epsCoeff=np.zeros(1),svCoeff=np.zeros(1),scaleUp = False,\
                            solids = 0.17,N=500,g=9.8,rouf = 1000.0,viscosity=0.00089,lamda = 0.5,plot = False
                            ):
        miu = frictionCoeff
        U_f = U
        Height = BedHeight
        # if not supplied, the following coefficients are default to compaction pressure measured from experiments
        if DensityCoeff.all() == 0:
            DensityCoeff = self.density.densityCoeff_Pc
        if epsCoeff.all() == 0:
            epsCoeff = self.porosity.porosityCoeff_Pc
        if svCoeff.all() == 0:
            svCoeff = self.modifiedKC.SvCoeff_Pc

        if scaleUp:
            tspan=np.linspace(0,Height,N)
        else:
            tspan=np.linspace(Height,0,N)
        c=[]

        def foce_balance_rhs(y,t): # y (sigma, in kPa), t ()
            yA = y
            density = Density.func_pressure(DensityCoeff,yA)
            eps = Porosity.func_porosity(epsCoeff,yA)
            m = solids*(1-eps)
            DGBouyantDh=m*(density/m-rouf)*g/1000# converted to kPa
            DfsDh = ModifiedKCAnalyzer.func_modifiedKC(svCoeff,U_f,epsCoeff,self.viscosity,ClassicKCAnalyzer.k,yA)
            DfwDh = (2*lamda*miu*yA/radius) # yA has unit of kPa
            if scaleUp:
                DsigmaDh = DfsDh - DGBouyantDh - DfwDh
            else:
                DsigmaDh = DfsDh - DGBouyantDh + DfwDh
            
            return DsigmaDh
        
        #sol = odeint(foce_balance_rhs,yinit,tspan,hmax=Height/(1*N))
        sol = odeint(foce_balance_rhs,yinit,tspan,hmax=Height/(1*N))
        sol = np.array(sol).flatten()

        sigma_abs = sol
        avgStress = np.sum(sigma_abs*Height/N)/Height
        densityProfile = Density.func_pressure(DensityCoeff,sigma_abs)
        densityAvg = np.sum(densityProfile*Height/N)/Height
        epsilonProfile = Porosity.func_porosity(epsCoeff,sigma_abs)
        epsilonAvg = np.sum(epsilonProfile*Height/N)/Height
        SvProfile = ModifiedKCAnalyzer.func_Sv(svCoeff,sigma_abs)
        dpdz = ModifiedKCAnalyzer.func_modifiedKC(svCoeff,U_f,epsCoeff,self.viscosity,ClassicKCAnalyzer.k,sigma_abs)

        if scaleUp:
            revDpDz=dpdz[::-1]
            revdeltaP2 = np.cumsum(revDpDz*Height/N)
            deltaP2=revdeltaP2[::-1]
        else:
            deltaP2 = np.cumsum(dpdz*Height/N)
        
        DfsDh = ModifiedKCAnalyzer.func_modifiedKC(svCoeff,U_f,epsCoeff,self.viscosity,ClassicKCAnalyzer.k,sigma_abs)
        DGBouyantDh = solids*(1-epsilonProfile)*(densityProfile/(solids*(1-epsilonProfile))-rouf)*g
        DfwDh = (2*lamda*miu*sigma_abs/radius)
        label = ["stress","density","epsilon","Sv","dpdz","deltaP","fs","GBouyant","fw"]
        return dict(height=tspan,stress =sigma_abs,avgStress=avgStress, density = densityProfile,avgDensity=densityAvg,\
                    epsilon=epsilonProfile,avgEps=epsilonAvg,Sv=SvProfile,dpdz = dpdz,deltaP = deltaP2,\
                    fs=DfsDh,GBouyant=DGBouyantDh,fw=DfwDh,label = label)
        

    def getProbLocIdx(self,N,Height):
        fp = self.exp.fp
        tspan=np.linspace(Height,0,N)
        dp2_low = fp.ele_ports[0]/1000 #unit: m
        dp2_high = fp.ele_ports[2]/1000
        idx_dp2low = np.where(tspan>dp2_low)[-1][-1]
        idx_dp2high = np.where(tspan<dp2_high)[0][0]
        dp3_low = fp.ele_ports[1]/1000 #unit: m
        dp3_high = fp.ele_ports[3]/1000
        idx_dp3low = np.where(tspan>dp3_low)[-1][-1]
        idx_dp3high = np.where(tspan<dp3_high)[0][0]
        return dp2_low,idx_dp2low,dp2_high,idx_dp2high,dp3_low,idx_dp3low,dp3_high,idx_dp3high

    def getModelDP(self,DP,N,Height):
        deltaP = DP
        dp2_low,idx_dp2low,dp2_high,idx_dp2high,dp3_low,idx_dp3low,dp3_high,idx_dp3high = self.getProbLocIdx(N,Height)
        Model_dp2 = (deltaP[idx_dp2low] - deltaP[idx_dp2high])/(dp2_high-dp2_low)
        Model_dp3 = (deltaP[idx_dp3low] - deltaP[idx_dp3high])/(dp3_high-dp3_low)
        Model_avgDP = np.average([Model_dp2,Model_dp3])
        Model_stdDP = np.std([Model_dp2,Model_dp3])
        return Model_dp2, Model_dp3, Model_avgDP,Model_stdDP

    def getModelProbeLocStress(self,Stress,N,Height):
        dp2_low,idx_dp2low,dp2_high,idx_dp2high,dp3_low,idx_dp3low,dp3_high,idx_dp3high = self.getProbLocIdx(N,Height)
        avgStress_dp2 = np.mean(Stress[idx_dp2high:idx_dp2low+1])
        stdStress_dp2 = np.std(Stress[idx_dp2high:idx_dp2low+1])
        avgStress_dp3 = np.mean(Stress[idx_dp3high:idx_dp3low+1])
        stdStress_dp3 = np.std(Stress[idx_dp3high:idx_dp3low+1])
        avgStress_avgDP = np.average([avgStress_dp2,avgStress_dp3])
        stdStress_avgDP = np.std([avgStress_dp2,avgStress_dp3])
        return avgStress_dp2,stdStress_dp2,avgStress_dp3,stdStress_dp3,avgStress_avgDP,stdStress_avgDP

    def compareModelExp(self,result,frictCoeff,Height,yinit,i_exp,rouCoeff=np.zeros(1),EpsCoeff=np.zeros(1),\
                        SvCoeff=np.zeros(1),rouf = 1000.0,viscosity=0.00089,plot = False):
        frictionCoeff = frictCoeff
        if rouCoeff.all() == 0:
            rouCoeff = self.density.densityCoeff
        if EpsCoeff.all() == 0:
            EpsCoeff = self.porosity.porosityCoeff
        if SvCoeff.all() == 0:
            SvCoeff = self.modifiedKC.SvCoeff
        fp = self.exp.fp    
        radius = fp.dia_col/2000 # unit: m
        
        DP_result = []
        N_Uf = len(result[i_exp]['avg_sfvel'])
        Model_dp2 = np.zeros(N_Uf)
        Model_dp3 = np.zeros(N_Uf)
        Model_avgDP = np.zeros(N_Uf)
        Model_stdDP = np.zeros(N_Uf)
        avgStress_dp2= np.zeros(N_Uf)
        stdStress_dp2= np.zeros(N_Uf)
        avgStress_dp3= np.zeros(N_Uf)
        stdStress_dp3= np.zeros(N_Uf)
        avgStress_avgDP= np.zeros(N_Uf)
        stdStress_avgDP= np.zeros(N_Uf)
        for i in range(N_Uf):
            U_f = result[i_exp]['avg_sfvel'][i] # unit: mm/s
            DP_result.append(self.getBedProperties(frictionCoeff,U_f,Height,radius,yinit,\
                                            DensityCoeff=rouCoeff,epsCoeff=EpsCoeff,\
                                            svCoeff = SvCoeff,rouf = rouf,viscosity=viscosity))
            DP = DP_result[i]['deltaP']
            Stress = DP_result[i]['stress']
            avgStress_dp2[i],stdStress_dp2[i],avgStress_dp3[i],stdStress_dp3[i],\
            avgStress_avgDP[i],stdStress_avgDP[i] = self.getModelProbeLocStress(Stress,500,Height)
            Model_dp2[i], Model_dp3[i],Model_avgDP[i],Model_stdDP[i] = self.getModelDP(DP,500,Height)

        exp_dp2 = result[i_exp]['pg_1']
        exp_std_dp2 = result[i_exp]['std_pg_1']
        exp_dp3 = result[i_exp]['pg_2']
        exp_std_dp3 = result[i_exp]['std_pg_2']
        exp_dp = np.concatenate((exp_dp2,exp_std_dp2,exp_dp3,exp_std_dp3))

        return dict(avgStressP1 = avgStress_dp2,stdStressP1 = stdStress_dp2,\
                    avgStressP2=avgStress_dp3,stdStressP2 = stdStress_dp3,avgStressDP=avgStress_avgDP,\
                    stdStressDP=stdStress_avgDP,Model_dpP1=Model_dp2,Model_dpP2=Model_dp3,\
                    Model_avgDP=Model_avgDP,Model_stdDP=Model_stdDP,BPData_U=DP_result)

    def CPBModelExpCompare(self,Pc,result,BedHeight,istart,rouCoefficient,EpsCoefficient,SvCoefficient,\
                       rouf = 1000.0,viscosity=0.00089,frictCoeff=0.10): #plot=True,save=False
        BP_ModelExp = []
        Model_dpP1 = []
        Model_dpP2 = []
        Model_avgDP = []
        Model_stdDP = []
        avgStressP1=[]
        avgStressP2=[]
        avgStressDP=[]
        stdStressDP=[]
        stdStressP1=[]
        stdStressP2=[]
        N_exp=len(Pc)
        for i_exp in range(istart,N_exp):
            yinit = [Pc[i_exp]]
            frictCoeff=0.10
            Height = BedHeight[i_exp]
            BP_ModelExp_U = self.compareModelExp(result,frictCoeff,Height,yinit,i_exp,rouCoeff=rouCoefficient,\
                                            EpsCoeff=EpsCoefficient,SvCoeff = SvCoefficient,rouf = rouf,viscosity=viscosity)
            BP_ModelExp.append(BP_ModelExp_U)
            Model_dpP1.append(BP_ModelExp_U['Model_dpP1'])
            Model_dpP2.append(BP_ModelExp_U['Model_dpP2'])
            Model_avgDP.append(BP_ModelExp_U['Model_avgDP'])
            Model_stdDP.append(BP_ModelExp_U['Model_stdDP'])
            avgStressP1.append(BP_ModelExp_U['avgStressP1'])
            stdStressP1.append(BP_ModelExp_U['stdStressP1'])
            avgStressP2.append(BP_ModelExp_U['avgStressP2'])
            stdStressP2.append(BP_ModelExp_U['stdStressP2'])
            avgStressDP.append(BP_ModelExp_U['avgStressDP'])
            stdStressDP.append(BP_ModelExp_U['stdStressDP'])
        
        return dict(i_init=istart,totalNo=N_exp,Model_dpP1=Model_dpP1,Model_dpP2=Model_dpP2,\
                    Model_avgDP=Model_avgDP,Model_stdDP=Model_stdDP,\
                    avgStressP1=avgStressP1,stdStressP1=stdStressP1,\
                    avgStressP2=avgStressP2,stdStressP2=stdStressP2,\
                    avgStressDP=avgStressDP,stdStressDP=stdStressDP,BP_ModelExp=BP_ModelExp)

    def get_avgPs(self,Pc,result,BedHeight,packingDensity,epsilon,frictCoeff = 0.10):
        """

        """
        radius = self.radius
        densityCoeff = self.density.densityCoeff_Pc
        EpsCoeff2 = self.porosity.porosityCoeff_Pc
        SvCoeff2 = self.modifiedKC.SvCoeff_Pc
        import matplotlib.pyplot as plt
        DP_U0 = []
        N_exp = len(Pc)
        avgPs=np.zeros(len(Pc))
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        for i_exp in range(1,N_exp):
            yinit = [Pc[i_exp]]
            # frictCoeff=0.10
            Height = BedHeight[i_exp]
            U_f = 0
            DP_U0.append(self.getBedProperties(frictCoeff,U_f,Height,radius,yinit,\
                                                DensityCoeff=densityCoeff,epsCoeff=EpsCoeff2,\
                                                svCoeff = SvCoeff2))
            avgPs[i_exp]=DP_U0[i_exp-1]['avgStress']
            labelstr = '%0.2f kPa avgStress=%0.2f' % (Pc[i_exp],DP_U0[i_exp-1]['avgStress'])
            ax.plot(DP_U0[i_exp-1]['height'],DP_U0[i_exp-1]['stress'],label=labelstr)
        ax.legend(loc='best')
        

        PsData=avgPs#pressure kPa
        rouData=np.array(packingDensity) #kg/m3
        x0_rou =  [1,1,1]
        # update P_s to calculate P_s based material properties

        #densityCoeffPs, _pressure_fitting_ps, _density_fitting_ps = self.density.model_fitting(x0_rou,PsData,rouData)
        self.density.update_Ps(PsData)
        densityCoeffPs = self.density.densityCoeff_Ps
        eps_x0 = [1,1]
        epsData=np.array(epsilon) #kg/m3
        #EpsCoeffPs, _pressure_fitting_ps, _porosity_fitting_ps = self.porosity.model_fitting(eps_x0,PsData,epsData)
        self.porosity.update_Ps(PsData)
        EpsCoeffPs = self.porosity.porosityCoeff_Ps
        x0_Sv = np.ones(2)*10
        #SvCoeffPs, _specificSurfaceArea, _Pdata, _pressure_fitting, _Sv_fitting= self.modifiedKC.fit_Sv(x_SvMKC_0=x0_Sv,
        #                                                                                                Pdata=PsData,EpsCoeff=EpsCoeffPs)
        self.modifiedKC.update_Ps(PsData)
        SvCoeffPs = self.modifiedKC.SvCoeff_Ps
        #modifiedKCValidation(EpsCoeffPs,SvCoeffPs,PsData,result)
    
        return DP_U0,avgPs,densityCoeffPs,EpsCoeffPs,SvCoeffPs
     
    def fitSvFromCPBModel(self,CPB_data,result,EpsCoeffPs,modifier=None,plot=True,save=False):
        import matplotlib.pyplot as plt
        avgStressDp2=CPB_data['avgStressP1']
        avgStressDp3=CPB_data['avgStressP2']
        N_Uf_total = 0
        istart=CPB_data['i_init']
        N_exp=CPB_data['totalNo']
        for i_exp in range(istart,N_exp):
            Udata=result[i_exp]['avg_sfvel'][1:] # Highest Pc,first zero velocity
            N_Uf_total = N_Uf_total+ len(Udata)
        ESSAModelData = []  
        Ps_Uf_AllData = []   
        #fig = plt.figure(figsize=(8, 6))
        #ax = fig.add_subplot(1, 1, 1)
        for i_exp in range(istart,N_exp):
            Udata=result[i_exp]['avg_sfvel'][1:] # Highest Pc,first zero velocity
            N_Uf =  len(Udata)    
            specificSurfaceArea1=np.zeros(N_Uf)
            specificSurfaceArea2=np.zeros(N_Uf)
            Ps_Uf_Data1 = np.zeros(N_Uf)
            Ps_Uf_Data2 = np.zeros(N_Uf)
            for i in range(1,N_Uf+1):
                Ps_Uf_Data1[i-1]=avgStressDp2[i_exp-1][i]
                Ps_Uf_Data2[i-1]=avgStressDp3[i_exp-1][i]
                eps_kc1=Porosity.func_porosity(EpsCoeffPs,Ps_Uf_Data1[i-1])
                eps_kc2=Porosity.func_porosity(EpsCoeffPs,Ps_Uf_Data2[i-1])
                Udata=result[i_exp]['avg_sfvel'][i] #mm/s  # here i+1 to skip first experiment
                ydata1=result[i_exp]['pg_1'][i]#kPa/m
                ydata2=result[i_exp]['pg_2'][i]#kPa/m
                x_Sv = np.ones(1)*40
                #res_lsq1=least_squares(funcMini2,x_Sv, loss='soft_l1', f_scale=0.1, args=(Udata,ydata1,eps_kc1))
                res_lsq1=least_squares(ClassicKCAnalyzer.minimize_func_KC,x_Sv, loss='soft_l1', f_scale=50,
                                        args=(Udata,ydata1,eps_kc1,self.viscosity,ClassicKCAnalyzer.k))
                specificSurfaceArea1[i-1]=res_lsq1.x[0]
                #print(specificSurfaceArea1)
                x_Sv = np.ones(1)*40
                res_lsq2=least_squares(ClassicKCAnalyzer.minimize_func_KC,x_Sv, loss='soft_l1', f_scale=0.1,
                                        args=(Udata,ydata2,eps_kc2,self.viscosity,ClassicKCAnalyzer.k))
                specificSurfaceArea2[i-1]=res_lsq2.x[0]
                
            ySvData1=specificSurfaceArea1
            ySvData2=specificSurfaceArea2
            ESSAModelData = np.concatenate((ESSAModelData,ySvData1,ySvData2))
            Ps_Uf_AllData = np.concatenate((Ps_Uf_AllData,Ps_Uf_Data1,Ps_Uf_Data2))

        PcPlot=np.linspace(0,40,1000)
        x0_Sv = np.array([40,0.02])
        #res_lsqSvPs2=least_squares(funcSvMini2,x0_Sv, loss='soft_l1', f_scale=0.1, args=(Ps_Uf_AllData,ESSAModelData))
        res_lsqSvPs2=least_squares(ModifiedKCAnalyzer.minimize_func_Sv,x0_Sv, loss='soft_l1', f_scale=0.1, 
                                    bounds=([10,0.001],[1000,0.03]), method='trf',args=(Ps_Uf_AllData,ESSAModelData))
        SvCoeffPs_Modeldp23 = res_lsqSvPs2.x

        if modifier != None:
            SvCoeffPs_Modeldp23 = SvCoeffPs_Modeldp23 + modifier
        
        if plot:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(1, 1, 1)
            
            plt.plot(Ps_Uf_AllData,ESSAModelData,'C3s',markersize = 12,label='Experiment dp2')
    #        plt.plot(PcPlot, funcSv2(SvCoeffPs_Modeldp23+[1.5,0.0090], PcPlot), 'C3-',label='Model',linewidth = 3)
            plt.plot(PcPlot, ModifiedKCAnalyzer.func_Sv(SvCoeffPs_Modeldp23+[13,0.009], PcPlot), 'C3-',label='Model',linewidth = 3)
            
            plt.legend(loc = 'upper right',prop={'size': 18, 'weight':'bold'},frameon=False,)
            ax.set_xlabel('$\mathregular{P_c}$ (kPa)')
            ax.set_ylabel('Effective specific surface area ($\mathregular{cm^2 cm^{-3}}$)')
            ax.tick_params(axis = 'y', which = 'major', direction = 'out', length = 6.0, width = 2.0)
            ax.tick_params(axis = 'x', which = 'major', direction = 'out', length = 6.0, width = 2.0)      
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(1.8)
            fig.tight_layout()
            plt.show()
        if save:
            fig.savefig('./asMillWM/asMillWM_SvEpsModel2_correct.jpg',dpi=300,bbox_inches='tight')
        return SvCoeffPs_Modeldp23

    def compute_Ps_related(self):
        Pdata = self.exp.Pc
        result = self.exp.result
        BedHeight = self.bedHeight
        packingDensity = self.packingDensity
        epsilon = self.exp.epsilon
        self.DP_U0, self.avgPs, self.densityCoeff_Ps, self.EpsCoeff_Ps, self.SvCoeff_Ps = self.get_avgPs(Pdata, result,BedHeight,
                                                                                                            packingDensity,epsilon)
        istart = 0
        densityCoeff = self.density.densityCoeff_Pc
        EpsCoeff = self.porosity.porosityCoeff_Pc
        SvCoeff = self.modifiedKC.SvCoeff_Pc
        self.CPB_Pc = self.CPBModelExpCompare(Pdata,result,BedHeight,istart,densityCoeff,EpsCoeff,SvCoeff) 
        EpsCoeff_Ps = self.EpsCoeff_Ps
        self.SvCoeffPs_Modeldp23 = self.fitSvFromCPBModel(self.CPB_Pc,result,EpsCoeff_Ps)
        self.CPB_Ps = self.CPBModelExpCompare(Pdata,result,self.bedHeight,istart,self.densityCoeff_Ps,self.EpsCoeff_Ps,
                                                self.SvCoeffPs_Modeldp23)
        
    def update_SvCoeffPs(self,modifier:np.ndarray):
        """
        modifier should be a 2D array
        """
        self.SvCoeffPs_Modeldp23 = self.fitSvFromCPBModel(self.CPB_Pc,self.exp.result,self.EpsCoeff_Ps,modifier)

        istart = 0
        self.CPB_Ps = self.CPBModelExpCompare(self.exp.Pc,self.exp.result,self.bedHeight,istart,self.densityCoeff_Ps,self.EpsCoeff_Ps,
                                                self.SvCoeffPs_Modeldp23)





        
#%%       
if __name__ == '__main__':
    localdbpath='./datapickles/'
    exp_path='./exp/'

    import glob
    import os
    filenames_abs = glob.glob(exp_path+'*.xlsx')
    nexp = len(filenames_abs)
    ## extract basename of files
    filenames = [] # basename of files (with extension)
    names = [] # names of experiments (no extension)
    for i in range(nexp):
        filename = os.path.basename(filenames_abs[i])
        filenames.append(filename)
        names.append(filename.split('.')[0])

    exp = ColumnExperiment(filenames_abs[1], localdbpath)
    density = Density(exp)
    porosity = Porosity(exp)
    exp.clean_data()
    modifiedKC = ModifiedKCAnalyzer(exp)

    packedbed = PackedBed(exp)
    modifier = [-16,0.009]
    packedbed.update_SvCoeffPs(modifier)
    CPB_Pc = packedbed.CPB_Pc
    CPB_Ps = packedbed.CPB_Ps
    Pc_fitting = packedbed.exp.Pc
    result_fitting = packedbed.exp.result
    compareCPBplot = ColumnPlots('CPB_Pc_plot')
    
    
    #compareCPBplot.plotCPBModelExpDPCompare(CPB_Pc,Pc_fitting,result_fitting,fn_str='PcCorelation')
    compareCPBplot.plotCPBModelExpDPCompare(CPB_Ps,Pc_fitting,result_fitting,fn_str='PsCorelation')

    

            



        
    
    