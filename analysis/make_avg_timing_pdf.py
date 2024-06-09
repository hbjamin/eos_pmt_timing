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
            #new_prob.append((prob[i-7]+prob[i-6]+prob[i-5]+prob[i-4]+prob[i-3]+prob[i-2]+prob[i-1]+prob[i]+prob[i+1]+prob[i+2]+prob[i+3]+prob[i+4]+prob[i+5]+prob[i+6]+prob[i+7])/15)
            new_tot+=new_prob[-1]
        else:
            new_prob.append(prob[i])
    final_tot=0
    for i in range(len(prob)):
        if time[i]>=start_time and time[i]<=end_time:
            new_prob[i]=new_prob[i]*tot/new_tot
            final_tot+=new_prob[i]
    return new_prob

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
    df=pd.read_csv('../pmt_final_test_info.csv',names=fields,index_col=False)

    # Loop through all the final test root files (created by analyzing raw h5 files)
    for i in range(len(df)):
        dir='/nfs/disk4/bharris/eos/eos_pmt_timing/berkeley_darkbox_pmt_data/'
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
                # Subtract pedestal (average counts between 0 and 30 ns)
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
        plt.figure(int(i*2+1))
        plt.plot(total_time[i][int(15/dt):int(25/dt)],total_counts[i][int(15/dt):int(25/dt)])
        plt.plot(x_fit,y_fit,color='red')
        plt.xlabel('deltat [ns]')
        plt.ylabel('Pulses/ns')
        plt.title(pmt_type[i])
        # Normalize PDF
        prob=normalize(total_counts[i],dt)
        total=0
        for j in range(len(prob)):
            total+=prob[j]*dt
        print('Sum of probability after normalization is:',total)
        ratio=getLatePulsing(total_time[i],total_counts[i],5*sigma)
        print('Late Ratio is:',100*ratio,'percent')
        prob=prob.tolist()
        time=total_time[i].tolist()
        print('time :',time)
        print('time_prob :',prob)
        plt.figure(int(i*2+2))
        plt.plot(time,prob)
        plt.yscale('log')
        plt.xlabel('Time [ns]')
        plt.ylabel('Probability')
        plt.xlim(-20,80)
        plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
