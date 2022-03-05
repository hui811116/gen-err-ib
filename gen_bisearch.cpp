#include <iostream>
#include <string>
#include "Eigen/Dense"
#include "unsupported/Eigen/MatrixFunctions"
#include <cstdlib>
#include <cstdio>
#include <iomanip>

using std::cout;
using std::endl;
//using Eigen::MatirxXf;
//using Eigen::VectorXf;

typedef Eigen::MatrixXd Mat;
typedef Eigen::VectorXd Vec;


typedef struct{ 
	bool converge;
	Mat pzcx;
} algout_t;


double calcMI(const Mat& pxy){
	Vec px_ric = (1.0/pxy.rowwise().sum().array()).matrix();
	Vec py_ric = (1.0/pxy.colwise().sum().array()).matrix();
	return ( pxy.array() * (px_ric.asDiagonal()*pxy*py_ric.asDiagonal()).array().log() ).sum();
}
double calcKL(const Mat& pxy1,const Mat& pxy2){
	return (pxy1.array() * (pxy1.array().log()-(pxy2.array()+1e-9).log())).sum();
}
double calcKL(const Vec& px,const Mat& py){
	return (px.array() * (px.array().log()-(py.array()+1e-9).log())).sum();
}

algout_t ibOrig(const Mat& pxy,size_t nz, double beta, double thres, size_t maxiter){
	Vec px = pxy.rowwise().sum();
	Vec py = pxy.colwise().sum();
	Mat pxcy = pxy * (1.0/py.array()).matrix().asDiagonal();
	Mat pycx = ((1.0/px.array()).matrix().asDiagonal() * pxy).transpose();
	// random initialization
	Mat pzcx = Mat::Random(nz,pxy.rows());
	pzcx.array() += 1.01;
	Vec tmpv = pzcx.colwise().sum();
	pzcx *= (1.0/tmpv.array()).matrix().asDiagonal();
	size_t itcnt = 0;
	bool conv_flag = false;
	Vec loglike(pxcy.cols());
	while(itcnt++ < maxiter){
		Vec pz = pzcx * px;
		Mat pycz = ((1.0/pz.array()).matrix().asDiagonal()*(pzcx*pxy)).transpose();
		Mat dkl = Mat(nz,pxy.rows());
		for(size_t xx=0;xx<pxy.rows();++xx){
			for(size_t zz=0;zz<nz;++zz){
				loglike = (pycx.col(xx).array().log()- (pycz.col(zz).array()+1e-7).log()).matrix();
				dkl(zz,xx) = (pycx.col(xx).array() * loglike.array()).sum();
			}
		}
		Mat new_pzcx = pz.asDiagonal() * ((-beta * dkl.array()).exp()).matrix();
		Vec tmpnorm = new_pzcx.colwise().sum();
		new_pzcx *= (1.0/tmpnorm.array()).matrix().asDiagonal();
		Vec dtv = (new_pzcx - pzcx).array().abs().matrix().colwise().sum();
		if( (dtv.array()< (2.0 * thres) ).all()){
			conv_flag = true;
			break;
		}else{
			pzcx = new_pzcx;
		}
	}
	
	
	// TODO:
	// return a tuple, with pzcx, niter, conv,
	// save the calculation of MI to outer functions...
	algout_t output = {.converge = conv_flag, .pzcx = pzcx};
	return output;
}

int main(){
	srand((unsigned int) time(0));
	Mat pycx(2,4);
	pycx << 0.90, 0.76, 0.15, 0.06,
			0.10, 0.24, 0.85, 0.94;
	Vec px(4);
	px << 0.25, 0.25, 0.25 ,0.25;
	Mat pxy = (pycx*px.asDiagonal()).transpose();
	cout<<"I(X;Y)="<<calcMI(pxy)<<endl;
	Vec py = pxy.colwise().sum();
	Mat pxcy = pxy * ((1.0/py.array()).matrix()).asDiagonal();


	// make sure the IB is implemented right...
	/*
	algout_t output = ibOrig(pxy,pycx.cols(),10.0,1e-6,10000);
	cout<<"output of IBorig"<<endl;
	cout<<"converged:"<<output.converge<<endl;
	cout<<"encoder probability:"<<endl;
	cout<<output.pzcx<<endl;
	*/
	/*
	double beta = 1.0;
	double beta_inc = 1.0;
	for(size_t ib = 0;ib<40;++ib){
		algout_t output = ibOrig(pxy,pycx.cols(),beta,1e-6,10000);
		Mat tmp_pzcy = output.pzcx * pxcy;
		double mizy = calcMI(tmp_pzcy * py.asDiagonal());
		double mizx = calcMI(output.pzcx * px.asDiagonal());
		cout<<beta<<","<<mizy<<","<<mizx<<","<<output.converge<<endl;
		beta += beta_inc;
	}
	*/
	
	double kl_thres = 1.0;

	double eps_crude[50];
	double eps_init = 0.01;
	double eps_step = 0.02;
	for(size_t t=0;t<50;++t){
		eps_crude[t] = eps_init;
		eps_init+= eps_step;
	}
	/*
	double eps_detail[20];
	double eps_start = -0.1;
	double eps_detail_step = 0.01;
	for(size_t t=0;t<20;++t){
		eps_detail[t] = eps_start;
		eps_start+= eps_detail_step;
	}*/

	// prepare the crude counter
	size_t eps_pycx_cnt[4];
	size_t eps_px_cnt[4];
	//std::memset(eps_pycx_cnt,0,sizeof(eps_pycx_cnt));
	//std::memset(eps_px_cnt,0,sizeof(eps_px_cnt));

	Mat crude_pycx(pycx.rows(),pycx.cols());
	Vec crude_px(pycx.cols());

	size_t nrun = 40;
	cout<<std::setw(6)<<"beta,"\
		<<std::setw(16)<<"IXY,"\
		<<std::setw(16)<<"kl_model,"\
		<<std::setw(16)<<"kl_train,"\
		<<std::setw(16)<<"kl_x,"\
		<<std::setw(16)<<"kl_y,";
	for(size_t tt=0;tt<4;++tt){
		cout<<std::setw(16)<<"kl_ycx_"<<tt;
		if(tt==3)
			cout<<endl;
		else
			cout<<",";
	}
	
	//cout<<"sizet:"<<sizeof(size_t)<<" but "<<sizeof(eps_px_cnt)<<endl;

	double beta = 1.0;
	double beta_inc = 0.5;
	for(size_t ib=0; ib<32; ib++){
		Mat best_pycx (pycx.rows(),pycx.cols());
		double best_mi = 0;
		for(size_t nn=0;nn<nrun;nn++){
			algout_t output = ibOrig(pxy,pycx.cols(),beta,1e-6,10000);
			// calculate the required metrics
			Mat tmp_pzy = output.pzcx *pxy;
			Vec pz_rci = (1.0/(tmp_pzy.rowwise().sum()).array()).matrix();
			Mat tmp_pycz = ( pz_rci.asDiagonal() * tmp_pzy   ).transpose();
			Mat tmp_pycx = tmp_pycz * output.pzcx;
			double tmp_mi = calcMI(tmp_pycx * px.asDiagonal());
			if(tmp_mi>best_mi){
				best_mi = tmp_mi;
				best_pycx = tmp_pycx;
			}
		}
		// FIXME: why best mi = 0?
		// now we have the best predictor now
		Mat best_pxy = (best_pycx * px.asDiagonal()).transpose();
		Vec best_py = best_pxy.colwise().sum();
		// perform a crude search
		// TODO: make this whole thing a function
		double crude_err = 0.0;
		Mat crude_eps_pycx (pycx.rows(),pycx.cols());
		Vec crude_eps_px (pycx.cols());
		// using carry algorithm
		std::memset(eps_pycx_cnt,0,sizeof(eps_pycx_cnt)); // reset 
		std::memset(eps_px_cnt,0,sizeof(eps_px_cnt));
		size_t carry =0;
		size_t locc = 1;
		size_t inner_locc = 1;
		size_t inner_carry = 0;
		size_t cnt_outer = 0;
		while(locc<4){
			cnt_outer++;
			// load the counters
			for(size_t ii=0;ii<4;++ii){
				//cout<<eps_pycx_cnt[ii]<<",";
				crude_pycx(0,ii) = eps_crude[eps_pycx_cnt[ii]];
				crude_pycx(1,ii) = 1.0 -crude_pycx(0,ii);
			}
			//cout<<"init:"<<crude_pycx<<endl;
			// do something
			// an inner loop for px...
			while(inner_locc<3){
				// load the inner counters
				//for(size_t cnt=0;cnt<4;++cnt)
				//	cout<<eps_px_cnt[cnt]<<",";
				//cout<<endl;
				for(size_t xx = 0;xx<3;++xx){
					crude_px(xx) = eps_crude[eps_px_cnt[xx]];
				}
				crude_px(3) = 1.0 - crude_px(Eigen::seq(0,2)).sum();
				// validity check
				if( (crude_px.array()<1.0).all() && (crude_px.array()>0.0).all()  ){
					// valid probability vector...
					Mat tmp_crude_pxy  = (crude_pycx * crude_px.asDiagonal()).transpose();
					// calculate the error
					//cout<<"calc_kl:"<<crude_pycx<<endl;
					double kl_model = calcKL(tmp_crude_pxy,best_pxy);
					double kl_train = calcKL(tmp_crude_pxy,pxy);
					if( (kl_train< kl_thres) &&  (kl_model>crude_err) ){
						crude_err = kl_model;
						crude_eps_pycx = crude_pycx;
						crude_eps_px = crude_px;
					}
				}
				// inner carry on
				eps_px_cnt[0] += 1;
				eps_px_cnt[0] %=50;
				inner_carry = (eps_px_cnt[0]==0)? 1 : 0;
				inner_locc = (eps_px_cnt[0]==0)? 1 : inner_locc;
				while(inner_carry!=0 && inner_locc<3){
					eps_px_cnt[inner_locc] += 1;
					eps_px_cnt[inner_locc] %= 50;
					inner_carry = (eps_px_cnt[inner_locc]==0)? 1 : 0;
					inner_locc += (eps_px_cnt[inner_locc]==0)? 1 : 0;
				}
			} // while(inner_locc<4)
			
			// carry on
			eps_pycx_cnt[0] += 1;
			eps_pycx_cnt[0]%= 50;
			carry = (eps_pycx_cnt[0]==0)? 1 : 0;
			locc = (eps_pycx_cnt[0]==0)? 1 : locc;
			while(carry!=0 && locc < 4){
				eps_pycx_cnt[locc]+=1;
				eps_pycx_cnt[locc]%=50;
				carry = (eps_pycx_cnt[locc]==0)? 1 : 0;
				locc += (eps_pycx_cnt[locc]==0)? 1 : 0;
			}
		} // while(locc<4)
		// now crude_err is the worst kl_model
		// crude_eps_pycx, crude_eps_px are corresponding conditional and marginal prob.
		// move on to detail search
		// do this again...
		/*
		double worst_err = 0;
		double worst_kl_train = 0;
		double worst_kl_x =0;
		double worst_kl_y = 0;
		Vec worst_kl_ycx (pycx.cols());
		Mat worst_eps_pycx (pycx.rows(),pycx.cols());
		Vec worst_eps_px (pycx.cols());
		// using carry algorithm
		std::memset(eps_pycx_cnt,0,sizeof(eps_pycx_cnt)); // reset 
		std::memset(eps_px_cnt,0,sizeof(eps_px_cnt));
		carry =0;
		locc = 1;
		inner_locc = 1;
		inner_carry = 0;
		while(locc<4){
			for(size_t ii=0;ii<4;++ii){
				crude_pycx(0,ii) = eps_detail[eps_pycx_cnt[ii]];
				crude_pycx(1,ii) = -eps_detail[eps_pycx_cnt[ii]];
			}
			crude_pycx = crude_eps_pycx + crude_pycx;
			// now we can check the validility first
			if( (crude_pycx.array() > 0.0).all() && (crude_pycx.array()<1.0 ).all() ){
				// do something
				while(inner_locc<4){
					// load the inner delta
					for(size_t jj=0;jj<3;++jj){
						crude_px(jj) = eps_detail[eps_px_cnt[jj]];
					}
					crude_px(3) = -crude_px(Eigen::seq(0,2)).sum();
					//cout<<"crude_eps"<<endl<<crude_eps_px<<endl;
					crude_px = crude_eps_px + crude_px;
					// check the validality again
					if( (crude_px.array()>0.0).all() && (crude_px.array()<1.0).all()){
						// now all is ready
						//cout<<"crude_px3,"<<endl<<crude_px<<endl;
						//cout<<"crude_pycx"<<endl<<crude_pycx<<endl;
						Mat detail_pxy = (crude_pycx * crude_px.asDiagonal()).transpose();
						Vec detail_py = detail_pxy.colwise().sum();
						//cout<<"detail_py"<<detail_py<<endl;
						double kl_model = calcKL(detail_pxy,best_pxy);
						double kl_train = calcKL(detail_pxy,pxy);
						if(kl_train<kl_thres && kl_model>worst_err ){
							// a valid divergence
							worst_err = kl_model;
							worst_kl_train = kl_train;
							worst_kl_x = calcKL(crude_px,px);
							worst_kl_y = calcKL(detail_py,best_py);
							worst_kl_ycx = (crude_pycx.array() * (crude_pycx.array().log()-(best_pycx.array()+1e-9).log())).matrix().colwise().sum();
							// copy the worst Mat and Vec?
							worst_eps_pycx = crude_pycx;
							worst_eps_px = crude_px;
						}else{
							// invalid divergence, skip
						} // if kl_train < kl_thres
					}else{
						// not valid, skip
					}// if crude_px 
					
					// update inner carry
					eps_px_cnt[0] += 1;
					eps_px_cnt[0] %= 20;
					inner_carry = (eps_px_cnt[0]==0)? 1:0;
					inner_locc = (eps_px_cnt[0]==0)? 1: inner_locc;
					while(inner_carry !=0 && inner_locc<4){
						eps_px_cnt[inner_locc] += 1;
						eps_px_cnt[inner_locc]%= 20;
						inner_carry = (eps_px_cnt[inner_locc]==0)? 1: 0;
						inner_locc += (eps_px_cnt[inner_locc]==0)? 1: 0;
					}
				}// while innerlocc
			}else{
				// not valid, do nothing
			} // if valid
			
			// update the carry 
			eps_pycx_cnt[0] += 1;
			eps_pycx_cnt[0]%= 20;
			carry = (eps_pycx_cnt[0]==0)? 1 : 0;
			locc = (eps_pycx_cnt[0]==0)? 1 : locc;
			while(carry!=0 && locc<4){
				eps_pycx_cnt[locc]+=1;
				eps_pycx_cnt[locc]%=20;
				carry = (eps_pycx_cnt[locc]==0)? 1 : 0;
				locc += (eps_pycx_cnt[locc]==0)? 1 : 0;
			}
		} // while(locc<4)
		*/
		// if no two way grid search
		Mat worst_pxy = (crude_eps_pycx * crude_eps_px.asDiagonal()).transpose();
		Vec worst_py = worst_pxy.colwise().sum();
		double worst_err = crude_err;
		double worst_kl_train = calcKL(worst_pxy,pxy);
		double worst_kl_x = calcKL(crude_eps_px,px);
		double worst_kl_y = calcKL(worst_py,best_py);
		Vec worst_kl_ycx = (crude_eps_pycx.array() * (crude_eps_pycx.array().log() - (best_pycx.array()+1e-9).log())).matrix().colwise().sum();
		// after two-way grid search, print the result
		cout<<std::setw(6)<<beta<<","\
		<<std::setw(16)<<best_mi<<","\
		<<std::setw(16)<<worst_err<<","\
		<<std::setw(16)<<worst_kl_train<<","\
		<<std::setw(16)<<worst_kl_x<<","\
		<<std::setw(16)<<worst_kl_y<<",";
		for(size_t tt=0;tt<4;++tt){
			cout<<std::setw(16)<<worst_kl_ycx(tt);
			if(tt!=3)
				cout<<",";
		}
		cout<<endl;
		beta += beta_inc;
	} // for(double beta=1.0.......)
	
	
	return 0;
}
