# Supplementary Software 
#
# for 
#
# "Calibration-on-the-spot": How to calibrate an EMCCD camera from its images
#
# by
#
# Kim I. Mortensen and Henrik Flyvbjerg
#
# Python (python.org) code for the calibration of an EMCCD camera
# from isolated diffraction-limited spots.
#
# This code is written for and tested with
#
# Python v. 2.7.9
# 
# and relies on the packages
#
# matplotlib v. 1.4.3
# scipy v. 0.15.1
# numpy v. 1.9.2
# PIL v. 1.1.7
#
# This code is licensed under a Creative Commons Attribution 4.0 International License
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/
#
# 19 May, 2016
#
# Kim I. Mortensen


from scipy.special import erfc
from pylab import *
import scipy.optimize
import numpy
import PIL
import os
import math


def erf(x):
    """ The error function. """
    return 1-erfc(x)

class ChiSquare:
    """ Class defining the chi-squared function to be minimized in GME."""

    def __init__(self,counts,a):
        self.counts=counts  #piel array
        self.a=a   # pixel width in nm
        self.npix=shape(counts)[0]   #number of pixels each side
        self.posvec=arange(-(self.npix-1.0)/2.0,(self.npix)/2.0,1.0)*a
        #print(self.posvec)

    def Value(self,x):

        counts=self.counts
        npix=self.npix
        posvec=self.posvec
                
        mux=x[0]
        muy=x[1]
        sigma=x[2]**2
        N=x[3]**2
        b=x[4]**2

        pij=zeros((npix,npix),float)
        for i in range(npix):
            for j in range(npix):
                pij[j,i]=0.25*(erf((posvec[i]-mux+self.a/2.0)/(sigma*sqrt(2)))\
                               -erf((posvec[i]-mux-self.a/2.0)/(sqrt(2)*sigma)))*\
                              (erf((posvec[j]-muy+self.a/2.0)/(sigma*sqrt(2)))\
                               -erf((posvec[j]-muy-self.a/2.0)/(sqrt(2)*sigma)))
        
        val=ravel(counts-N*pij-b)
        return val


class CotsFIT:
    """ Class for tracking spots using GME and calibrating an EMCCD camera using COTS."""

    def __init__(self,fitframes,pw,\
                 initvals,initpix,deltapix,fit_image_stack):

        # Store the numpy array of images
        self.fit_image_stack=fit_image_stack

        # Store current working directory
        self.cwd=os.getcwd()

        self.fitframes=fitframes
        self.nfitframes=len(fitframes)
        
        # Store user input
        self.pw=pw
        self.deltapix=deltapix
        self.npix=2*deltapix

        # Store initial values for localization analysis
        self.initvals=initvals
        self.initpix=initpix

        # Set initial frame number
        self.frame=0

        # Create arrays for results
        self.counts=array([])
        self.expectedcounts=array([])
        
        self.Gest=zeros(len(fitframes))
        self.S0pest=zeros(len(fitframes))
        
        self.varG=zeros(len(fitframes))
        self.varS0p=zeros(len(fitframes))
        self.covarGS0p=zeros(len(fitframes))
                                    
        self.deltaG=zeros(len(fitframes))
        self.deltaS0p=zeros(len(fitframes))

        self.fij=zeros(len(fitframes)*self.npix**4)
        
        # Array for storage of localization estimates
        self.locest=zeros((len(fitframes),8))
        # Array for storing goodness of fit
        self.r_squared=0
        self.r_squared_array=zeros(len(fitframes))




    def SetFrame(self,frame):
        """ Sets the current frame number."""
        self.frame=frame
        
        
    def Estimate(self,frame=None):
        """ Localizes using GME."""

        self.SetFrame(frame)

        ypix=self.initpix[0]      # Initial pixel centering
        xpix=self.initpix[1]
        deltapix=self.deltapix

        # Load entire data matrix from specified frame in file
        counts=self.fit_image_stack[self.frame,:,:]
        
        # Extract pixel array around initial pixel
        counts=counts[ypix-deltapix:ypix+deltapix,xpix-deltapix:xpix+deltapix]

        #initvals = pylab.array([mux_init,muy_init,sigma_init,N_init,b_init])


        # Transformation of initial values
        pinit=zeros(5)
        pinit[0:2]=self.initvals[0:2]
        pinit[2]=sqrt(self.initvals[2])
        pinit[3]=sqrt(self.initvals[3])
        pinit[4]=sqrt(self.initvals[4])
        
        #print ('\nLocalization, frame number:'+str( self.frame))

        # Create instance of ChiSquare object
        cs=ChiSquare(counts,self.pw)

        # Localization with GME
        res,cov,infodict,mesg,ie =scipy.optimize.leastsq(cs.Value,pinit,ftol=0.0001,full_output=True)

        ss_err=(infodict['fvec']**2).sum()
        ss_tot=((counts-counts.mean())**2).sum()
        r_squared=1-(ss_err/ss_tot)
        
        lsest=res

        # Convert estimates
        est=zeros(5)
        est[0]=lsest[0]         #Position x (in nm)
        est[1]=lsest[1]         #Position y (in nm)
        est[2]=lsest[2]**2      #Std deviation
        est[3]=lsest[3]**2      #Power under dist
        est[4]=lsest[4]**2      #Bias
        
        self.mux=est[0]         #Store current position
        self.muy=est[1]
        self.est=concatenate((array([ypix,xpix]),array(est)))

        self.r_squared=r_squared

        return


    def Track(self):
        """ Tracks a spot by localizing it in each frame using GME."""

        # Loop over relevant frames
        for nframe in range(len(self.fitframes)):
            frame=self.fitframes[nframe]
            print(nframe)

            notfitted=True
            tracking_counter = 0
            while notfitted:
                print('in while')
                self.SetFrame(frame)
                self.Estimate(frame)

                if abs(self.muy/self.pw)<1.0 and abs(self.mux/self.pw)<1.0:
                    notfitted=False
                 #   print("Fitted")
                   # print(self.initpix)

                    self.locest[nframe,:]=append(frame,self.est)
                    self.r_squared_array[nframe]=self.r_squared
                  #  print(self.est)

                else:
                    print("Correcting initial pixel")
                    tracking_counter = tracking_counter + 1
                    
                    if(tracking_counter == 20):
                        notfitted=False


                    # If estimated position outside bounds of center pixel, shift center pixel in that direction

                    self.initpix+=array([int(round(self.muy/self.pw)),int(round(self.mux/self.pw))])
                    print(self.initpix)

        return



    def Signals(self):
        """ Calculates the fitted expected pixel output-signals."""
            
        # Loop over relevant frames
        for nframe in range(self.nfitframes):
            frame=self.fitframes[nframe]

            # Get localization estimates for relevant frame number
            estimates=self.locest[nframe,:]

            try:
                ypix=int(estimates[1])
                xpix=int(estimates[2])
                mux=estimates[3]
                muy=estimates[4]
                k=estimates[5]
                N=estimates[6]
                b=estimates[7]
            except TypeError:
                return

            # Load the raw signal values from file
            counts=self.fit_image_stack[frame,:,:]
            
            # Get the relevant subset of the pixel array
            deltapix=self.deltapix
            
            counts=counts[ypix-deltapix:ypix+deltapix, xpix-deltapix:xpix+deltapix]

            npix=shape(counts)[0]
            posvec=arange(-(npix-1.0)/2.0,npix/2.0,1.0)*self.pw

            # Calculate the fitted expected output-signal values
            pex=zeros((npix,npix))
            for i in range(npix):
                for j in range(npix):
                    pex[j,i]=0.25*(erf((posvec[i]-mux+self.pw/2.0)/(k*sqrt(2)))-\
                                    erf((posvec[i]-mux-self.pw/2.0)/(sqrt(2)*k)))*\
                                    (erf((posvec[j]-muy+self.pw/2.0)/(k*sqrt(2)))\
                                    -erf((posvec[j]-muy-self.pw/2.0)/(sqrt(2)*k)))

            S=N*pex+b

            # Store signals and expected signals for calibration
            self.counts=append(self.counts,ravel(counts))
            self.expectedcounts=append(self.expectedcounts,ravel(S))

        return
        
    def TransferMatrix(self):
        """ Calculates values of the transfer matrix."""

        npix=2*self.deltapix
        npixsq=npix**2
        
        # Loop over frames in directory
        for nframe in range(self.nfitframes):
            frame=self.fitframes[nframe]

            # Get estimates for chosen frame number
            estimates=self.locest[nframe]

            try:
                mux=estimates[3]
                muy=estimates[4]
                k=estimates[5]
                N=estimates[6]
                b=estimates[7]

            except TypeError:
                return

            # Select expected counts
            expectedcounts=self.expectedcounts[nframe*npixsq:(nframe+1)*npixsq]

            # Probability for recording a signal in pixel number i
            probi=(expectedcounts-b)/N
            
            # Calculate pixel distances from true center
            posvec=arange(-(npix-1.0)/2.0,npix/2.0,1.0)*self.pw
            ri=zeros((npix,npix))
            xi=zeros((npix,npix))
            yi=zeros((npix,npix))
            for i in range(npix):
                for j in range(npix):
                    risq=(posvec[j]-muy)**2+(posvec[i]-mux)**2
                    ri[j,i]=sqrt(risq)
                    xi[j,i]=(posvec[i]-mux)
                    yi[j,i]=(posvec[j]-muy)

            # Coerce distances into vector
            ri=ravel(ri)
            xi=ravel(xi)
            yi=ravel(yi)

            # Derivatives of expectation values (mux,muy,N,b,sigma)
            Eip=zeros((npix**2,5))
            Eip[:,0]=N*xi/k**2*probi
            Eip[:,1]=N*yi/k**2*probi
            Eip[:,2]=probi
            Eip[:,3]=ones(npix**2)
            Eip[:,4]=N*(ri**2/k**3-2.0/k)*probi

            # 'Information' matrix for Gaussian mask
            I=array([
            [sum(Eip[:,0]*Eip[:,0]),sum(Eip[:,0]*Eip[:,1]),sum(Eip[:,0]*Eip[:,2]),sum(Eip[:,0]*Eip[:,3]),sum(Eip[:,0]*Eip[:,4])],\
                [sum(Eip[:,1]*Eip[:,0]),sum(Eip[:,1]*Eip[:,1]),sum(Eip[:,1]*Eip[:,2]),sum(Eip[:,1]*Eip[:,3]),sum(Eip[:,1]*Eip[:,4])],\
                [sum(Eip[:,2]*Eip[:,0]),sum(Eip[:,2]*Eip[:,1]),sum(Eip[:,2]*Eip[:,2]),sum(Eip[:,2]*Eip[:,3]),sum(Eip[:,2]*Eip[:,4])],\
                [sum(Eip[:,3]*Eip[:,0]),sum(Eip[:,3]*Eip[:,1]),sum(Eip[:,3]*Eip[:,2]),sum(Eip[:,3]*Eip[:,3]),sum(Eip[:,3]*Eip[:,4])],\
                [sum(Eip[:,4]*Eip[:,0]),sum(Eip[:,4]*Eip[:,1]),sum(Eip[:,4]*Eip[:,2]),sum(Eip[:,4]*Eip[:,3]),sum(Eip[:,4]*Eip[:,4])]])

            # Inverse 'information' matrix
            Iinv=inv(I)
            
            fij=zeros((npix**2,npix**2))
            
            for j in range(npix**2):

                # Transfer matrix
                fi=zeros(npix**2)
                Eja=Eip[j,:]
                for i in range(npix**2):
                    Eib=Eip[i,:] # vector of length 5
                    fi[i]=dot(Eja,dot(Iinv,Eib))

                fij[:,j]=fi

            self.fij[nframe*npixsq**2:(nframe+1)*npixsq**2]=ravel(fij)

        return
        
        

    def Calibrate(self,Ginit,S0init):
        """ Performs calibration using COTS."""
        
        npixsq=self.npix**2

        # Perform COTS for each frame
        for nframe in range(len(self.fitframes)):
            
            #print("\nCalibration, frame number:"+str(self.fitframes[nframe])+"\n")
            S=self.counts[nframe*npixsq:(nframe+1)*npixsq]
            E=self.expectedcounts[nframe*npixsq:(nframe+1)*npixsq]
            
            def likelihood(pars):
                """ Function defining the negative log-likelihood to be minimized in COTS."""
                G=pars[0]
                GS0=pars[1]
                sigmasq=2*(G*E-GS0)
                negll=sum(log(sigmasq)+(S-E)**2/sigmasq)
                return negll
                        
            (Gest_temp,S0est_temp)=scipy.optimize.fmin(likelihood,(Ginit,Ginit*S0init),ftol=1e-8,disp=False)
            self.Gest[nframe]=Gest_temp
            self.S0pest[nframe]=S0est_temp
        
        # Calculation of bias and covariance matrix for estimates for each frame
        for nframe in range(len(self.fitframes)):
            
            # Get the fitted expected pixel output-signals
            Etrue=self.expectedcounts[nframe*npixsq:(nframe+1)*npixsq]
            
            G=mean(self.Gest)
            S0p=mean(self.S0pest)
            S0=S0p/G
            
            v=2*G*(Etrue-S0)
            
            # Get the transfer function for the relevant frame
            f=self.fij[nframe*npixsq**2:(nframe+1)*npixsq**2]
            f=reshape(f,(npixsq,npixsq))
            
            # Calculate the information matrix for the parameters
            IGG=-sum(Etrue**2/v**2)+4*sum(Etrue**2/v**2*diag(f))
            IGS0p=-sum(Etrue/v**2)+4*sum(Etrue/v**2*diag(f))
            IS0pS0p=-sum(1/v**2)+4*sum(1/v**2*diag(f))
            for i in range(npixsq):
                for j in range(npixsq):
                    temp=(-2*f[i,j]**2*v[j]/v[i]**3)
                    IGG+=Etrue[i]**2*temp
                    IGS0p+=Etrue[i]*temp
                    IS0pS0p+=temp
                    
            IGG*=4
            IGS0p*=-4
            IS0pS0p*=4
            
            detI=IGG*IS0pS0p-IGS0p**2
                        
            # Calculate auxiliary quantities
            aux_a=sum(1.0/v**2)
            aux_b=sum(Etrue/v**2)
            aux_c=2*sum(Etrue/v)
            aux_d=-2*sum(1.0/v)
            aux_e=4*sum(Etrue**2/v**2)
            aux_f=4*aux_a
            aux_i=-4*aux_b
            aux_k=aux_e/4
            
            aux_g=0.0
            aux_h=0.0
            aux_j=0.0
            for i in range(npixsq):
                for j in range(npixsq):
                    if j!=i:
                        aux_g+=Etrue[i]*Etrue[j]/((Etrue[i]-S0)*(Etrue[j]-S0))
                        aux_h+=1.0/((Etrue[i]-S0)*(Etrue[j]-S0))
                        aux_j+=Etrue[i]/((Etrue[i]-S0)*(Etrue[j]-S0))
            aux_g/=G**2
            aux_h/=G**2
            aux_j/=(-G**2)
            
            # Calculate theoretical covariance matrix
            self.varG[nframe]=1.0/detI**2*(IS0pS0p**2*(-aux_c**2+3*aux_e+aux_g)+IGS0p**2*(-aux_d**2+3*aux_f+aux_h)
                                -2*IS0pS0p*IGS0p*(-aux_c*aux_d+3*aux_i+aux_j))
                                                        
            self.varS0p[nframe]=1.0/detI**2*(IGS0p**2*(-aux_c**2+3*aux_e+aux_g)+IGG**2*(-aux_d**2+3*aux_f+aux_h)
                                -2*IGG*IGS0p*(-aux_c*aux_d+3*aux_i+aux_j))
                                        
            self.covarGS0p[nframe]=1.0/detI**2*(-IS0pS0p*IGS0p*(-aux_c**2+3*aux_e+aux_g)-IGG*IGS0p*(-aux_d**2+3*aux_f+aux_h)
                            +(IS0pS0p*IGG+IGS0p**2)*(-aux_c*aux_d+3*aux_i+aux_j))
            
            # Calculate additional auxiliary quantities
            aux_l=2*sum(Etrue*diag(f)/v)
            aux_n=-2*sum(diag(f)/v)
            aux_m=0.0
            aux_o=0.0
                
            for j in range(npixsq):
                for i in range(npixsq):
                    aux_m+=(2*Etrue[i]/v[i]**2*v[j]*f[i,j]**2)
                    aux_o+=(-2/v[i]**2*v[j]*f[i,j]**2)
            
            # Calculate theoretical bias            
            self.deltaG[nframe]=1/detI*(IS0pS0p*(2*aux_l-aux_m)-IGS0p*(2*aux_n-aux_o))
            self.deltaS0p[nframe]=1.0/detI*(-IGS0p*(2*aux_l-aux_m)+IGG*(2*aux_n-aux_o))
            
        nframes=len(self.fitframes)
        
        # Calculate average parameter estimates
        Gav=mean(self.Gest-self.deltaG)
        Gav_var=mean(self.varG)/nframes
        S0pav=mean(self.S0pest-self.deltaS0p)
        S0pav_var=mean(self.varS0p)/nframes
                
        S0est=(self.S0pest-self.deltaS0p)/Gav
        varS0=((self.S0pest-self.deltaS0p)/Gav**2)**2*Gav_var+(1.0/Gav)**2*self.varS0p\
            -2*((self.S0pest-self.deltaS0p)/Gav**2)*(1.0/Gav)*self.covarGS0p/nframes
        
        GS0pav_covar=mean(self.covarGS0p)/nframes
        S0av_var=(S0pav/Gav**2)**2*Gav_var+(1.0/Gav)**2*S0pav_var-2*(S0pav/Gav**2)*(1.0/Gav)*GS0pav_covar
        
        print ('\nAverage gain = '+str(Gav)+ '+/-'+str( sqrt(Gav_var)))
        print ('Average offset = '+str(S0pav/Gav)+ '+/-'+str( sqrt(S0av_var))+"\n")
                        
        return self.Gest-self.deltaG, self.S0pest-self.deltaS0p, self.locest, self.r_squared_array, self.varG
    
    
def fit_cots(fitframes, pw, initvals, initpix, deltapix, image_stack, g_init, s0_init):
    """ Function to package up all steps of Mortensen Implementation
    """
    
    cots=CotsFIT(fitframes, pw, initvals, initpix, deltapix, image_stack)

    # Track the spot using GME
    cots.Track()

    ## Calculate the fitted expected pixel output-signal values
    cots.Signals()

    ## Calculate the values of the transfer function
    cots.TransferMatrix()

    ## Calibrate using COTS
    Gest, S0pest, locest, rsquared, Gvar=cots.Calibrate(g_init, s0_init)

    return Gest, S0pest, locest, rsquared, Gvar