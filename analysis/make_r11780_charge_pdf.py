import uproot
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# You will probably delete these (except for timing_res)
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

def main():

	parser=argparse.ArgumentParser()
	parser.add_argument('file')
	args=parser.parse_args()
	f=args.file
	print('Opening File:',f)
	
	with uproot.open(f) as file:
		
		# Get ROOT file
		print(file.keys())
		leaf=file['charge']

		# Get time of prompt peak
		counts = leaf.values()
		bin_edges = leaf.axes[0].edges()
		time=(bin_edges[:-1]+bin_edges[1:])/2
		
		# Keep  0.3 pC < data < 3 pC
		# cut counts and time array

		max_index=getMaxIndex(counts)
		time_of_max=raw_starttime+max_index*timing_res
		print('Time of prompt peak:',time_of_max)
		
		# Fit peak to gaussian to get TTS sigma and plot to 3sigma
		fit_start_time=time_of_max-10
		fit_end_time=time_of_max+10
		
		params,covariance=fitPeakToGaussian(time,counts)
		errors=np.sqrt(np.diag(covariance))
		A=params[0]
		B=params[1]
		C=params[2]
		A_err=errors[0]
		B_err=errors[1]
		C_err=errors[2]	
	
		print('Fit a TTS sigma of',C,'+/-',C_err)
		x_fit=np.linspace(B-2*C,B+2*C,100)
		y_fit=gaussian(x_fit,A,B,C)
		plt.figure(1)
		plt.plot(time,counts)
		plt.plot(x_fit,y_fit,color='red')
		plt.xlabel('deltat [ns]')
		plt.ylabel('Pulses/ns')
		plt.xlim(40,60)	

		# Change this to center peak at 1.6 pC
		#time,counts=recenter(counts,max_index,final_starttime,final_endtime)
		#ratio=getLatePulsing(time,counts,5*C)
	
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
