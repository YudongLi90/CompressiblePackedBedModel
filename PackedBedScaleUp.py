import numpy as np
from ClassicKCAnalyzer import ClassicKCAnalyzer
from ColumnExperiment import ColumnExperiment
from ColumnPlots import ColumnPlots
from Density import Density
from ModifiedKCAnalyzer import ModifiedKCAnalyzer
from Porosity import Porosity
from scipy.integrate import odeint

class PackedBedScaleUp:
    """
    a packed bed class without functionalities of fitting Sv from Ps. This is only used for directly simulating the packed bed properties
    along the bed height direction using experimentally fitted ModifiedKC equation parameters (KCL equation).
    This is developed for the air classified material permeability experiment to show that how different material affect reactor scale up
    """
    def __init__(self, exp=None,density=None,porosity=None,modifiedKC=None, Uf=1, height=1.0, 
                radius=0.5, yinit=0, frictionCoeff=0.1, scaleUp=True,plot=False):
        if isinstance(exp,ColumnExperiment):
            self.exp = exp
        else:
            raise ValueError("Please provide valide ColumnExperiment object to instantiate.")
        if isinstance(density,Density): # recommend use cleaned data (exp) (manual cleaned more prefered)
            self.density = density
        else: # you might want to use density before data clean up (but need to be careful to check whether density have outliers)
            self.density = Density(exp)
        if isinstance(porosity, Porosity):
            self.porosity = porosity
        else:
            self.porosity = Porosity(exp)
        if isinstance(modifiedKC,ModifiedKCAnalyzer):
            self.modifiedKC = modifiedKC
        else:
            self.modifiedKC = ModifiedKCAnalyzer(exp,self.porosity)
        self.densityCoeff = density.densityCoeff_Pc
        self.porosityCoeff = self.porosity.porosityCoeff_Pc
        self.SvCoeff = self.modifiedKC.SvCoeff_Pc

        self.Uf = Uf # in mm/s
        self.height = height
        self.radius = radius
        self.yinit = yinit
        self.frictionCoeff = frictionCoeff
        # parameters not likely to be changed (can be changed through update_scaleup)
        self.solids = 0.17
        self.N = 500
        self.g = 9.8
        self.rouf = 1000
        self.viscosity = 0.00089
        self.lamda = 0.5
        self.plot = plot

    
    def update_scaleup(self,**kwargs):
        for key,value in kwargs.items():
            if hasattr(self,key):
                setattr(self,key,value)
        self.scaleup()

    def scaleup(self):
        return self.getBedProperties(frictionCoeff=self.frictionCoeff,U=self.Uf,BedHeight=self.height,radius=self.radius,yinit=self.yinit,
                                DensityCoeff=self.densityCoeff,epsCoeff=self.porosityCoeff,svCoeff=self.SvCoeff,scaleUp=True,
                                solids=self.solids,N=self.N,g=self.g,rouf=self.rouf,viscosity=self.viscosity,lamda=self.lamda,plot=self.plot)


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
        result =  dict(height=tspan,stress =sigma_abs,avgStress=avgStress, density = densityProfile,avgDensity=densityAvg,\
                    epsilon=epsilonProfile,avgEps=epsilonAvg,Sv=SvProfile,dpdz = dpdz,deltaP = deltaP2,\
                    fs=DfsDh,GBouyant=DGBouyantDh,fw=DfwDh,label = label)
        if plot:
            scaleupPlot = ColumnPlots('dirtyDryexperiments')
            scaleupPlot.plot_scaleup(scaleUpData=[result],variable=[self.exp.name])
        return result