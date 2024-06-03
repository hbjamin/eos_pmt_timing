import uproot
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


##########################################################################
# Hardcoded global variables that set the timing scale and window of the PDF
raw_starttime=0 #ns
raw_endtime=160 #ns
timing_res=0.1 #ns
final_starttime=-20 #ns
final_endtime=80 #ns
nbins=int((raw_endtime-raw_starttime)/timing_res)
##########################################################################

def normalizePDF(data):
        sum=0
        for i in range(len(data)):
                sum+=data[i]
        data=data/sum
        return data

def getMaxIndex(data):
        max=0
        max_index=0
        for i in range(len(data)):
                if data[i]>max:
                        max=data[i]
                        max_index=i
        return max_index

def recenter(data,max_index,final_starttime,final_endtime):
        new_time=[]
        new_data=[]
        counter=0
        for i in range(len(data)):
                if i>=max_index+final_starttime/timing_res and i<max_index+final_endtime/timing_res:
                        new_data.append(data[i])
                        new_time.append(round(final_starttime+timing_res*counter,1))
                        counter+=1
        return new_time,new_data

def subtractPedestal(time,data,ped_start_time,ped_end_time):
	ped=0
	counts=0
	for i in range(len(time)):
		if time[i]>=ped_start_time and time[i]<=ped_end_time:
			ped+=data[i]
			counts+=1
	ped=ped/counts
	print('Pedestal:',ped)
	for i in range(len(data)):
		data[i]=data[i]-ped
		if data[i]<0:
			data[i]=0
	return data

def gaussian(x,A,B,C):
    return A*np.exp(-((x-B)**2)/(2*C**2))

def fitPeakToGaussian(time,data):
	max_index=getMaxIndex(data)
	A_guess=data[max_index]
	B_guess=time[max_index]
	C_guess=1
	fit_time=[]
	fit_data=[]
	for i in range(len(data)):
		if i>max_index-2*C_guess/timing_res and i<max_index+2*C_guess/timing_res:
			fit_time.append(time[i])
			fit_data.append(data[i])
	params,covariance=curve_fit(gaussian,fit_time,fit_data,p0=[A_guess,B_guess,C_guess])
	return params,covariance

def getLatePulsing(time,data,time_cut):
	early=0
	early_count=0
	late=0
	late_count=0
#	print('***********')
#	print('Calculating Late Ratio.....')
#	print('Time cut is',time_cut)
	for i in range(len(data)):
		if time[i]<time_cut:
			early+=data[i]
			early_count+=1
		else:
			late+=data[i]
			late_count+=1
#	early=early/early_count
#	late=late/late_count
#	print('Total early is',early)
#	print('Total late is',late)
	fraction = late/(early+late)
	print('Late Ratio:',round(100*fraction,2))
#	print('**********')
	return fraction

def averageWithNearestNeighbors(time,data,start_time,end_time):
#	print('Smoothing from',start_time,'to',end_time)
	old_sum=0
	for i in range(len(data)):
		if time[i]>=start_time and time[i]<=end_time:
			old_sum+=data[i]
	new_sum=0
	new_data=[]
	for i in range(len(data)):
		if time[i]>=start_time and time[i]<=end_time:
			new_data.append((data[i-7]+data[i-6]+data[i-5]+data[i-4]+data[i-3]+data[i-2]+data[i-1]+data[i]+data[i+1]+data[i+2]+data[i+3]+data[i+4]+data[i+5]+data[i+6]+data[i+7])/15)
			#new_data.append((data[i-4]+data[i-3]+data[i-2]+data[i-1]+data[i]+data[i+1]+data[i+2]+data[i+3]+data[i+4])/9)
			#new_data.append((data[i-3]+data[i-2]+data[i-1]+data[i]+data[i+1]+data[i+2]+data[i+3])/7)
			#new_data.append((data[i-2]+data[i-1]+data[i]+data[i+1]+data[i+2])/5)
			#new_data.append((data[i-1]+data[i]+data[i+1])/3)
			new_sum+=new_data[-1]
		else:
			new_data.append(data[i])
#	print('old:',old_sum)
#	print('new:',new_sum)
	print(len(data),len(new_data))
	final_sum=0
	for i in range(len(data)):
		if time[i]>=start_time and time[i]<=end_time:
			new_data[i]=new_data[i]*old_sum/new_sum
			final_sum+=new_data[i]
#	print('final',final_sum)
	return new_data

def main():

	parser=argparse.ArgumentParser()
	parser.add_argument('file')
	args=parser.parse_args()
	f=args.file
	print('Opening File:',f)
	
	with uproot.open(f) as file:
		
		# Get ROOT file
		print(file.keys())
		leaf=file['time']

		# Get time of prompt peak
		counts = leaf.values()
		bin_edges = leaf.axes[0].edges()
		time=(bin_edges[:-1]+bin_edges[1:])/2
		max_index=getMaxIndex(counts)
		time_of_max=time[max_index]
		print('Time of prompt peak:',time_of_max)
		
		# Fit peak to gaussian to get TTS sigma and plot to 3sigma
		fit_start_time=time_of_max-10
		fit_end_time=time_of_max+10
		nbins_fit=int((fit_end_time-fit_start_time)/timing_res)
		
		params,covariance=fitPeakToGaussian(time,counts)
		errors=np.sqrt(np.diag(covariance))
		A=params[0]
		B=params[1]
		C=params[2]
		A_err=errors[0]
		B_err=errors[1]
		C_err=errors[2]	
	
		print('Fit a TTS sigma of',round(C,3),'+/-',round(C_err,3))
		x_fit=np.linspace(B-2*C,B+2*C,100)
		y_fit=gaussian(x_fit,A,B,C)
		plt.figure(1)
		plt.plot(time,counts)
		plt.plot(x_fit,y_fit,color='red')
		plt.xlabel('deltat [ns]')
		plt.ylabel('Pulses/ns')
		plt.xlim(fit_start_time,fit_end_time)	
	
		# Smooth out late pulsing
		counts=averageWithNearestNeighbors(time,counts,time[max_index]+5*C,time[-10])
		# Smooth out pre pulsing
		counts=averageWithNearestNeighbors(time,counts,time[10],time[max_index]-3*C)

		# Get nhits at every deltat to make a PDF
		plt.figure(2)
		plt.plot(time,counts)
		plt.yscale('log')
		plt.xlabel('deltat [ns]')
		plt.ylabel('Pulses/ns')	
	
		max_index=getMaxIndex(counts)
		# Subtract pedestal
		print('Subtracting pedestal...')
		counts=subtractPedestal(time,counts,0,30)
	
		# Center prompt peak at 0 and shorten window
		print('Recentering...') 
		time,counts=recenter(counts,max_index,final_starttime,final_endtime)
		
		ratio=getLatePulsing(time,counts,5*C)
	
		# Normalize PDF
		prob=normalizePDF(counts)
		prob=prob.tolist()
		print('time :',time)
		print('time_prob :',prob)
		plt.figure(3)
		plt.plot(time,prob)
		plt.yscale('log')
		plt.xlabel('Time [ns]')
		plt.ylabel('Probability')
		plt.show()

if __name__ == '__main__':
	main()
