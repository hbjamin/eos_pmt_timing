import os
import math
import uproot
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Normalize probability by area
def normalize(prob,dt):
        tot=0
        for i in range(len(prob)):
                tot+=prob[i]
        prob=prob/tot/dt
        return prob

# Return index with largest probability
def getMaxIndex(prob):
        val=0
        max_i=0
        for i in range(len(prob)):
                if prob[i]>val:
                        val=prob[i]
                        max_i=i
        return max_i

# Define time domain of pdf and recenter it so that largest probability is at 0 ns
def recenter(prob,max_i,time_domain,dt):
        new_time=[]
        new_prob=[]
        n=0
        for i in range(len(prob)):
                if i>=max_i+time_domain[0]/dt and i<max_i+time_domain[1]/dt:
                        new_prob.append(prob[i])
                        new_time.append(round(time_domain[0]+dt*n,1))
                        n+=1
        new_prob=normalize(new_prob,dt)
        return new_time,new_prob

# Remove dark pulsing pedestal
def subtractPedestal(time,prob,ped_start_time,ped_end_time):
        ped=0
        n=0
        for i in range(len(time)):
                if time[i]>=ped_start_time and time[i]<=ped_end_time:
                        ped+=prob[i]
                        n+=1
        ped=ped/n
        # Don't allow negative probability
        for i in range(len(prob)):
                prob[i]=prob[i]-ped
                if prob[i]<0:
                        prob[i]=0
        return prob

def gaussian(t,A,mu,sigma):
    return A*np.exp(-((t-mu)**2)/(2*sigma**2))

def exponential_func(t,A,b,c):
    return A*np.exp(b*(t-c))

def mirrored_exponential_func(t,A,b,c):
    return A*np.exp(-b*(t-c))

def fitPeakToGaussian(time,prob,dt):
    max_i=getMaxIndex(prob)
    A_guess=prob[max_i]
    mu_guess=time[max_i]
    sigma_guess=1
    fit_time=[]
    fit_prob=[]
    for i in range(len(prob)):
        if i>max_i-2*sigma_guess/dt and i<max_i+2*sigma_guess/dt:
            fit_time.append(time[i])
            fit_prob.append(prob[i])
    params,covariance=curve_fit(gaussian,fit_time,fit_prob,p0=[A_guess,mu_guess,sigma_guess])
    return params,covariance

# Return fraction of probability after cuttoff time
def getLatePulsing(time,prob,cutoff_time):
    early=0
    late=0
    for i in range(len(prob)):
        if time[i]<cutoff_time:
            early+=prob[i]
        else:
            late+=prob[i]
    fraction = late/(early+late)
    return fraction

# Smooth section of pdf by averaging with n nearest neightbors to the left and right (should probably be done by convoling with a gaussian instead)
def averageWithNearestNeighbors(time,prob,start_time,end_time,n):
    tot=0
    for i in range(len(prob)):
        if time[i]>=start_time and time[i]<=end_time:
            tot+=prob[i]
    new_tot=0
    new_prob=[]
    for i in range(len(prob)):
        if time[i]>=start_time and time[i]<=end_time:
            avg=0
            for j in range(2*n+1):
                avg+=prob[i-n+j]/(2*n+1)
            new_prob.append(avg)
            new_tot+=new_prob[-1]
        else:
            new_prob.append(prob[i])
    final_tot=0
    for i in range(len(prob)):
        if time[i]>=start_time and time[i]<=end_time:
            new_prob[i]=new_prob[i]*tot/new_tot
            final_tot+=new_prob[i]
    return new_prob


# Find intersection of exponential fit of start of prompt peak and average
# dark rate before prompt peak. Then remove everything before that 
def removeNoiseBeforePromptPeak(time,prob):
    # Get dark rate
    dark_rate=n=0
    for i in range(len(time)):
        if time[i]<-10:
            dark_rate+=prob[i]
            n+=1
    dark_rate=dark_rate/n
    # Get data to fit 
    fit_time=[]
    fit_prob=[]
    for i in range(len(time)):
        if time[i]>=-3 and time[i]<=-1:
            fit_time.append(time[i])
            fit_prob.append(prob[i])
    # Make initial guess of parameters and fit
    guess=[0.005,0.5,-3] 
    params,covariance=curve_fit(exponential_func,fit_time,fit_prob,guess)
    y_fit=exponential_func(time,params[0],params[1],params[2]) 
    plt.plot(time,y_fit,color='red')
    x_line=np.linspace(-20,0,100)
    y_line=[]
    for i in range(len(x_line)):
        y_line.append(dark_rate)
    plt.plot(x_line,y_line,color='red')
    # Find intersection
    for i in range(len(time)):
        if y_fit[i]>dark_rate:
            # Remove everything before that 
            print("crossing at",time[i])
            return time[i:],prob[i:]

# Find intersection of exponential fit of tail of late pulsing peak
# and average dark rate after. Then remove everything after that
# **** This is only done for 8" PMTs because of their weird problem ****
def removeNoiseAfterLatePeak(time,prob):
    # Get dark rate
    dark_rate=n=0
    for i in range(len(time)):
        if time[i]>50 and time[i]<60:
            dark_rate+=prob[i]
            n+=1
    dark_rate=dark_rate/n
    # Get data to fit 
    fit_time=[]
    fit_prob=[]
    for i in range(len(time)):
        if time[i]>=37 and time[i]<=40:
            fit_time.append(time[i])
            fit_prob.append(prob[i])
    # Make initial guess of parameters and fit
    guess=[0.005,0.1,35] 
    params,covariance=curve_fit(mirrored_exponential_func,fit_time,fit_prob,guess)
    y_fit=mirrored_exponential_func(time,params[0],params[1],params[2]) 
    plt.plot(time,y_fit,color='red')
    x_line=np.linspace(35,60,100)
    y_line=[]
    for i in range(len(x_line)):
        y_line.append(dark_rate)
    plt.plot(x_line,y_line,color='red')
    # Find intersection
    for i in range(len(time)):
        if y_fit[-1-i]>dark_rate:
            print("crossing at",time[i])
            return time[:-1-i],prob[:-1-i]
    return time,prob

def main():
    
    # Hardcoded variables for timing resolution and domain of pdf
    dt=0.1 # timing resolution in ns
    old_time_domain=[0,160] # ns
    new_time_domain=[-20,80] # ns
    nbins=int((old_time_domain[1]-old_time_domain[0])/dt)

    # Vectors to store all of the results for a given pmt type. Used to create average timing pdf across desired timing domain
    pmt_type=['R14688','R7081','R11780']
    cols=int((new_time_domain[1]-new_time_domain[0])/dt) 
    total_counts=np.zeros((len(pmt_type),cols))
    total_time=np.zeros((len(pmt_type),cols))
    
    # Get the information about each pmt's final test 
    fields=['serial','type','test_id','gain','path']
    df=pd.read_csv('pmt_final_test_info.csv',names=fields,index_col=False)

    # Loop through all the final test root files (created by analyzing raw h5 files)
    for i in range(len(df)):
        dir='berkeley_darkbox_pmt_data/'
        file = dir+df['path'][i]
        if os.path.exists(file):
            with uproot.open(file) as f:
                # Get the delta t histogram
                hist=f['time']
                # Fit the peak to a gaussian
                counts=hist.counts()
                bin_edges=hist.axes[0].edges()
                time=(bin_edges[:-1]+bin_edges[1:])/2
                params,covariance=fitPeakToGaussian(time,counts,dt)
                errors=np.sqrt(np.diag(covariance))
                sigma=params[2]
                sigma_err=errors[2]
                max_index=getMaxIndex(counts)
                
                # Smooth out late pulsing (pulses after 5 sigma of prompt peak)
                counts=averageWithNearestNeighbors(time,counts,time[max_index]+5*sigma,time[-10],7)       
                
                # Smooth out pre pulsing (pulses before 3 sigma of promt peak)
                counts=averageWithNearestNeighbors(time,counts,time[10],time[max_index]-3*sigma,7)
                
                # Subtract pedestal 
                # At this point the window is -50 to ~150 ns
                # 0 to 30 ns is always dark pulses for all runs of all pmt types
                counts=subtractPedestal(time,counts,0,30)   
                
                # Center prompt peak at 0 ns
                time,counts=recenter(counts,max_index,new_time_domain,dt)
                ratio=getLatePulsing(time,counts,5*sigma)
                # Get the pmt type
                current_pmt_type=0
                if (df['type'][i]=='R7081'):
                    current_pmt_type=1     
                elif(df['type'][i]=='R11780'):
                    current_pmt_type=2
                print('Seial:',df['serial'][i],'--- Type:',df['type'][i],'--- TTS sigma:',round(sigma,2),'+/-',round(sigma_err,2),'--- TTS FWHM:',round(2*sigma*math.sqrt(2*math.log(2)),2),'+/-',round(2*sigma_err*math.sqrt(2*math.log(2)),2),'--- Late Ratio:',round(100*ratio,2),'%')
                # Add results to vector for each pmt type
                for i in range(len(counts)):
                    total_counts[current_pmt_type][i]+=counts[i]
                    total_time[current_pmt_type][i]=time[i]
        else:
            print('Cannot find:',file)     
    # For every pmt type
    for i in range(len(pmt_type)):  
        print('Average',pmt_type[i],'timing pdf')
        # Normalize PDF
        total_counts[i]=normalize(np.array(total_counts[i]),dt).tolist()
        # Fit the peak to a gaussian
        params,covariance=fitPeakToGaussian(total_time[i],total_counts[i],dt)
        errors=np.sqrt(np.diag(covariance))
        A=params[0]
        mu=params[1]
        sigma=params[2]
        sigma_err=errors[2]
        print('Fit a TTS sigma of',round(sigma,4),'+/-',round(sigma_err,4))
        x_fit=np.linspace(mu-2*sigma,mu+2*sigma,100)
        y_fit=gaussian(x_fit,A,mu,sigma) 
        #plt.figure(int(i*2+1))
        #plt.plot(total_time[i][int(15/dt):int(25/dt)],total_counts[i][int(15/dt):int(25/dt)])
        #plt.plot(total_time[i],total_counts[i])
        #plt.plot(x_fit,y_fit,color='red')
        #plt.xlabel('deltat [ns]')
        #plt.ylabel('Pulses/ns')
        #plt.title(pmt_type[i])



      

        # Cut out all pulses after late pulsing
        # See function definition for details
        # See docdb documentation for reasoning

        
        plt.figure(int(i*2+2))
        plt.plot(total_time[i],total_counts[i],color='blue')
        # Remove dark rate before prompt peak
        time_new,prob_new=removeNoiseBeforePromptPeak(total_time[i],total_counts[i])
        prob_new=normalize(np.array(prob_new),dt).tolist()
        time_new=time_new.tolist()
        if i==0:
            # Remove dark rate after late peak 
            time_new,prob_new=removeNoiseAfterLatePeak(time_new,prob_new)
        plt.plot(time_new,prob_new,color='orange')
        plt.yscale('log')
        plt.ylim(1e-6,1)
        plt.xlabel('Time [ns]')
        plt.ylabel('Probability')
        plt.xlim(-20,80)
        plt.grid()

        total=0
        for j in range(len(prob_new)):
            total+=prob_new[j]*dt
        print('Sum of probability after normalization is:',total)
        ratio=getLatePulsing(time_new,prob_new,5*sigma)
        print('Late Ratio is:',100*ratio,'percent')
        #prob_new=prob_new.tolist()
        #time_new=time_new.tolist()
        print('time :',time_new)
        print('time_prob :',prob_new)
    plt.show()

if __name__ == '__main__':
    main()
