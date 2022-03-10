import numpy as np
import sys
import os
import matplotlib.pyplot as plt


args = sys.argv

threshold = float(args[2])
result_dict = {'beta_res':[],'pycx_train':None,'px_train':None}

with open(args[1],"r") as fid:
	allraw_lines = fid.readlines()
	all_lines = [item.replace("  "," ") for item in allraw_lines]
	idx = 0
	while idx < len(all_lines):
		# preprocess line
		# NOTE: beta must proceed data
		#print('start',all_lines[idx])
		if "[beta]" in all_lines[idx]:
			idx +=1
			tmp_beta = float(all_lines[idx].strip())
			tmp_res = {'beta':tmp_beta,'best_pycx':None,'pycx_eps':None,'px_eps':None}
			idx+=1
		elif "[pycx_train]" in all_lines[idx]:
			idx += 1
			tmp_mat = []
			while idx+1<len(all_lines):
				line_ele = [float(item) for item in all_lines[idx].strip().split()]
				tmp_mat.append(line_ele)
				idx+=1
				if "[" in all_lines[idx]:
					# end of parsing
					#tmp_pycx_train = np.array(tmp_map)
					result_dict['pycx_train'] = np.array(tmp_mat)
					#result_dict[tmp_beta]['pycx_train'] = np.array(tmp_map)
					break
		elif "[px_train]" in all_lines[idx]:
			idx+=1
			tmp_vec = []
			while idx+1<len(all_lines):
				element = float(all_lines[idx].strip())
				tmp_vec.append(element)
				idx+=1
				if "[" in all_lines[idx]:
					#result_dict[tmp_beta]['px_train'] = np.array(tmp_vec)
					result_dict['px_train'] = np.array(tmp_vec)
					break
		elif "[best_pycx]" in all_lines[idx]:
			idx+=1
			tmp_mat = []
			while idx+1 < len(all_lines):
				#print("$$",all_lines[idx],"$$")
				#print(all_lines[idx].strip().split(" "))
				line_ele = [float(item) for item in all_lines[idx].strip().split()]
				tmp_mat.append(line_ele)
				idx+=1
				if "[" in all_lines[idx]:
					tmp_res['best_pycx'] = np.array(tmp_mat)
					break
		elif "[eps_px]" in all_lines[idx]:
			# this is the last before next beta goes
			idx+=1
			tmp_vec = []
			while idx < len(all_lines):
				element = float(all_lines[idx].strip())
				tmp_vec.append(element)
				idx+=1
				if idx >= len(all_lines):
					tmp_res['px_eps'] = np.array(tmp_vec)
					result_dict['beta_res'].append(tmp_res)
					break
				elif "[" in all_lines[idx]:
					tmp_res['px_eps'] = np.array(tmp_vec)
					result_dict['beta_res'].append(tmp_res)
					break
		elif "[eps_pycx]" in all_lines[idx]:
			idx+=1
			tmp_mat=[]
			while idx+1 < len(all_lines):
				line_ele = [float(item) for item in all_lines[idx].strip().split()]
				tmp_mat.append(line_ele)
				idx+=1
				if "[" in all_lines[idx]:
					tmp_res['pycx_eps'] = np.array(tmp_mat)
					break
		else:
			print('untracked line:{}'.format(all_lines[idx]))
			idx+=1


def calcMI(pxy):
	px = np.sum(pxy,axis=1)
	py = np.sum(pxy,axis=0)
	return np.sum(pxy*np.log(pxy/px[:,None]/py[None,:]))
def calcKL(pxy1,pxy2):
	return np.sum(pxy1*(np.log(pxy1)-np.log(pxy2)))

train_pycx = result_dict['pycx_train']
train_px   = result_dict['px_train']
train_pyx = train_pycx * train_px[None,:]
train_py = np.sum(train_pyx,axis=1)
mi_train = calcMI(train_pycx*train_px[None,:])
print('IXY_train,{:10.5f}'.format(mi_train))
integrate_results = []
print('{:8s},{:10s},{:10s},{:10s},{:10s},{:10s},{:10s},{:10s},{:10s},{:10s},{:10s}'.format(
	'beta','mi_model','mi_eps','kl_model','kl_train','kl_eps_x','kl_eps_y','kl_model_y',
	'var_model_hycx','theory','py',
	))
for item in result_dict['beta_res']:
	beta = item['beta']
	best_pycx = item['best_pycx']
	best_pyx = best_pycx*train_px[None,:]
	best_py = np.sum(best_pyx,axis=1)
	eps_px = item['px_eps']
	eps_pycx = item['pycx_eps']
	eps_pyx = eps_pycx * eps_px[None,:]
	eps_py = np.sum(eps_pyx,axis=1)
	# metrics that are needed

	best_mi = calcMI(best_pyx)
	eps_mi = calcMI(eps_pyx)
	# all kld
	kl_train = calcKL(eps_pyx,train_pyx)
	kl_model = calcKL(eps_pyx,best_pyx)
	kl_eps_x = calcKL(eps_px,train_px)
	kl_eps_y = calcKL(eps_py,best_py)
	kl_model_y = calcKL(best_py,train_py)
	# we also need the entropy vector
	model_hycx = np.sum(-best_pycx*np.log(best_pycx),axis=0)
	# what matter is the sample variance of this vector
	var_model_hycx = np.var(model_hycx)

	# estimated bound
	thebd = 0.5* var_model_hycx + len(best_py)*(1/np.exp(1)) + 2*threshold - kl_model_y
	# pythagorean bound
	pybd = np.abs(best_mi-eps_mi) + kl_eps_x+kl_eps_y

	integrate_results.append([beta,best_mi,eps_mi,kl_model,kl_train,kl_eps_x,kl_eps_y,kl_model_y,var_model_hycx,thebd,pybd])
	print('{:8.4f},{:10.5f},{:10.5f},{:10.5f},{:10.5f},{:10.5f},{:10.5f},{:10.5f},{:10.5f},{:10.5f},{:10.5f}'.format(
			beta,best_mi,eps_mi,kl_model,kl_train,kl_eps_x,kl_eps_y,kl_model_y,var_model_hycx,thebd,pybd
		))

np_results =np.array(integrate_results)