#include <iostream>
#include <iterator>
#include <string>
#include "Eigen/Dense"
#include "unsupported/Eigen/MatrixFunctions"
#include <cstdlib>
#include <cstdio>
#include <iomanip>
#include <fstream>
#include <boost/program_options.hpp>

namespace po = boost::program_options;
using std::cout;
using std::endl;
using std::cerr;

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

int main(int ac, char** av){
	Mat pycx;
	Vec px;
	double kl_thres,beta,beta_inc;               // change the threshold as constraints
	size_t nbeta;
	try{
		double thres,bstep;
		int nbe;
		std::string dsel;
		po::options_description desc("Allowed options");
		desc.add_options()
			("help","display available options")
			("threshold",po::value<double>(&thres)->default_value(1.0), "set the constraint threshold")
			("betastep",po::value<double>(&bstep)->default_value(1.0), "set the value for beta increment")
			("betanum",po::value<int>(&nbe)->default_value(20), "number of beta for sweeping")
			("dataset",po::value<std::string>(&dsel)->default_value("sim23"), "choose from [sim23,sim24]")
		;

		po::variables_map vm;
		po::store(po::parse_command_line(ac,av,desc),vm);
		po::notify(vm);

		if (vm.count("help")){
			cout<<desc<<endl;
			return 0;
		}
		
		// thresholds of KL
		if (vm.count("threshold")){
			cout<<std::setw(30)<<std::left<<"constraint threshold:"
				<< std::setw(10)<<std::right<<vm["threshold"].as<double>() << endl;
		}else{
			cout<< "default threshold:" << std::setw(10)<<std::right<<vm["threshold"].as<double>() << endl;
		}
		kl_thres = vm["threshold"].as<double>();
		// beta increment
		cout<<std::setw(30)<<std::left<<"beta increment:"
			<<std::setw(10)<<std::right<<vm["betastep"].as<double>() << endl;
		beta_inc = vm["betastep"].as<double>();
		
		// number of beta
		if (vm.count("betanum")){
			cout<<std::setw(30)<<std::left<<"# of beta:"
				<<std::setw(10)<<std::right<<vm["betanum"].as<int>()<<endl;
		} else{
			cout<<std::setw(30)<<std::left<<"default # of beta:"
				<<std::setw(10)<<std::right<<vm["betanum"].as<int>()<<endl;
		}
		nbe = vm["betanum"].as<int>();
		nbeta = (nbe>0)? (size_t)nbe : 1;

		// datasets to run
		if (vm.count("dataset")){
			cout<<std::setw(30)<<std::left<<"set data to:"
				<<std::setw(10)<<std::right<<vm["dataset"].as<std::string>()<< endl;

		}else{
			cout<<std::setw(30)<<std::left<<"default data:"
				<<std::setw(10)<<std::right<<vm["dataset"].as<std::string>()<<endl;
		}

		std::string arg_data = vm["dataset"].as<std::string>();
		if (arg_data.compare("sim24")==0){
			pycx.resize(2,4);
			pycx << 0.90, 0.76, 0.15, 0.06,
				0.10, 0.24, 0.85, 0.94;
			px.resize(4);
			px << 0.25, 0.25, 0.25 ,0.25;
		}else if(arg_data.compare("sim23")==0){
			pycx.resize(2,3);
			pycx << 0.90, 0.76, 0.06,
					0.10, 0.24, 0.94;
			px.resize(3);
			px << 0.33, 0.34, 0.33;
		}else{
			cerr<<"undefined dataset, abort..."<<endl;
			return 1;
		}		

	}catch(std::exception& e){
		cerr<< "error: "<<e.what()<<endl;
		return 1;
	}catch(...){
		cerr<< "Exception of unknown type!"<<endl;
	}

	srand((unsigned int) time(0));
	

	Mat pxy = (pycx*px.asDiagonal()).transpose();
	cout<<"pycx"<<endl<<pycx<<endl;
	cout<<"px"<<endl<<px<<endl;
	cout<<"I(X;Y)="<<calcMI(pxy)<<endl;
	Vec py = pxy.colwise().sum();
	Mat pxcy = pxy * ((1.0/py.array()).matrix()).asDiagonal();

	size_t xdim = pycx.cols();

	// make sure the IB is implemented right...
	
	
	size_t crude_len = 20;
	double eps_crude[crude_len];
	double eps_init = 0.01;
	double eps_step = 0.05;
	for(size_t t=0;t<crude_len;++t){
		eps_crude[t] = eps_init;
		eps_init+= eps_step;
	}
	
	size_t detail_len = 20;
	double eps_detail[detail_len];
	double eps_start = -0.1;
	double eps_detail_step = 0.01;
	for(size_t t=0;t<detail_len;++t){
		eps_detail[t] = eps_start;
		eps_start+= eps_detail_step;
	}

	// prepare the crude counter
	size_t eps_pycx_cnt[xdim];
	size_t eps_px_cnt[xdim-1];  // dof = |X|-1, since the last one is 1-sum(xi)
	//std::memset(eps_pycx_cnt,0,sizeof(eps_pycx_cnt));
	//std::memset(eps_px_cnt,0,sizeof(eps_px_cnt));

	Mat crude_pycx(pycx.rows(),xdim);
	Vec crude_px(xdim);

	size_t nrun = 40;
	cout<<std::setw(5)<<"beta,"\
		<<std::setw(12)<<"IXY_t,"\
		<<std::setw(12)<<"IXY_e,"\
		<<std::setw(12)<<"kl_model,"\
		<<std::setw(12)<<"kl_train,"\
		<<std::setw(12)<<"kl_x,"\
		<<std::setw(12)<<"kl_y,";
	for(size_t tt=0;tt<xdim;++tt){
		cout<<std::setw(12)<<"kl_ycx_"<<tt;
		if(tt==xdim-1)
			cout<<endl;
		else
			cout<<",";
	}

	// prepare the IO head
	std::fstream fid;
	fid.open( "gen_bisearch_output.txt",std::fstream::out);
	fid<<"[pycx_train]"<<endl<<pycx<<endl;
	fid<<"[px_train]"<<endl<<px<<endl;

	
	beta= 1.0;
	for(size_t ib=0; ib<nbeta; ib++){
		Mat best_pycx (pycx.rows(),xdim);
		double best_mi = 0;
		for(size_t nn=0;nn<nrun;nn++){
			algout_t output = ibOrig(pxy,xdim,beta,1e-6,10000);
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
		Mat crude_eps_pycx (pycx.rows(),xdim);
		Vec crude_eps_px (xdim);
		// using carry algorithm
		//std::memset(eps_pycx_cnt,0,sizeof(eps_pycx_cnt)); // reset 
		for(size_t ss = 0;ss<xdim;ss++)
			eps_pycx_cnt[ss] = 0;
		size_t carry =0;
		size_t locc = 1;
		size_t inner_locc = 1;
		size_t inner_carry = 0;
		while(locc<xdim){
			// load the counters
			for(size_t ii=0;ii<xdim;++ii){
				crude_pycx(0,ii) = eps_crude[eps_pycx_cnt[ii]];
				crude_pycx(1,ii) = 1.0 -crude_pycx(0,ii);
			}
			// do something
			// an inner loop for px...
			// reset the inner loop!!!
			inner_locc = 1;
			inner_carry=0;
			//std::memset(eps_px_cnt,0,sizeof(eps_px_cnt));
			for(size_t rs=0;rs<xdim-1;++rs)
				eps_px_cnt[rs] = 0;
			while(inner_locc<xdim-1){
				// load the inner counters
				for(size_t xx = 0;xx<xdim-1;++xx){
					crude_px(xx) = eps_crude[eps_px_cnt[xx]];
				}
				crude_px(xdim-1) = 1.0 - crude_px(Eigen::seq(0,xdim-2)).sum();
				// validity check
				if( (crude_px.array()<1.0).all() && (crude_px.array()>0.0).all()  ){
					// valid probability vector...
					Mat tmp_crude_pxy  = (crude_pycx * crude_px.asDiagonal()).transpose();
					// calculate the error
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
				eps_px_cnt[0] %= crude_len;
				inner_carry = (eps_px_cnt[0]==0)? 1 : 0;
				inner_locc = (eps_px_cnt[0]==0)? 1 : inner_locc;
				while( (inner_carry!=0) && (inner_locc<xdim-1)  ){
					eps_px_cnt[inner_locc] += 1;
					eps_px_cnt[inner_locc] %= crude_len;
					inner_carry = (eps_px_cnt[inner_locc]==0)? 1 : 0;
					inner_locc += (eps_px_cnt[inner_locc]==0)? 1 : 0;
				}
			} // while(inner_locc<4)
			
			// carry on
			eps_pycx_cnt[0] += 1;
			eps_pycx_cnt[0]%= crude_len;
			carry = (eps_pycx_cnt[0]==0)? 1 : 0;
			locc = (eps_pycx_cnt[0]==0)? 1 : locc;
			while( (carry!=0) && (locc < xdim) ){
				eps_pycx_cnt[locc]+=1;
				eps_pycx_cnt[locc]%=crude_len;
				carry = (eps_pycx_cnt[locc]==0)? 1 : 0;
				locc += (eps_pycx_cnt[locc]==0)? 1 : 0;
			}
		} // while(locc<4)
		// now crude_err is the worst kl_model
		// crude_eps_pycx, crude_eps_px are corresponding conditional and marginal prob.
		// move on to detail search
		// do this again...
		
		double worst_err = 0;
		double worst_kl_train = 0;
		double worst_kl_x =0;
		double worst_kl_y = 0;
		double worst_mi_eps = 0;
		Vec worst_kl_ycx (xdim);
		Mat worst_eps_pycx (pycx.rows(),xdim);
		Vec worst_eps_px (xdim);
		// using carry algorithm
		//std::memset(eps_pycx_cnt,0,sizeof(eps_pycx_cnt)); // reset 
		for(size_t rr =0;rr<xdim;rr++)
			eps_pycx_cnt[rr] = 0;
		carry =0;
		locc = 1;
		while(locc<xdim){
			for(size_t ii=0;ii<xdim;++ii){
				crude_pycx(0,ii) = eps_detail[eps_pycx_cnt[ii]];
				crude_pycx(1,ii) = -eps_detail[eps_pycx_cnt[ii]];
			}
			crude_pycx = crude_eps_pycx + crude_pycx;
			// now we can check the validility first
			if( (crude_pycx.array() > 0.0).all() && (crude_pycx.array()<1.0 ).all() ){
				// do something
				inner_locc = 1;
				inner_carry = 0;
				//std::memset(eps_px_cnt,0,sizeof(eps_px_cnt));
				for(size_t rr=0;rr<xdim-1;++rr)
					eps_px_cnt[rr] = 0;
				while(inner_locc<xdim-1){
					// load the inner delta
					for(size_t jj=0;jj<xdim-1;++jj){
						crude_px(jj) = eps_detail[eps_px_cnt[jj]];
					}
					crude_px(xdim-1) = -crude_px(Eigen::seq(0,xdim-2)).sum();
					crude_px = crude_eps_px + crude_px;
					// check the validality again
					if( (crude_px.array()>0.0).all() && (crude_px.array()<1.0).all()){
						// now all is ready
						Mat detail_pxy = (crude_pycx * crude_px.asDiagonal()).transpose();
						Vec detail_py = detail_pxy.colwise().sum();
						double kl_model = calcKL(detail_pxy,best_pxy);
						double kl_train = calcKL(detail_pxy,pxy);
						if(kl_train<kl_thres && kl_model>worst_err ){
							// a valid divergence
							worst_err = kl_model;
							worst_kl_train = kl_train;
							worst_kl_x = calcKL(crude_px,px);
							worst_kl_y = calcKL(detail_py,best_py);
							worst_kl_ycx = (crude_pycx.array() * (crude_pycx.array().log()-(best_pycx.array()+1e-9).log())).matrix().colwise().sum();
							worst_mi_eps = calcMI(crude_pycx * crude_px.asDiagonal());
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
					eps_px_cnt[0] %= detail_len;
					inner_carry = (eps_px_cnt[0]==0)? 1: 0;
					inner_locc = (eps_px_cnt[0]==0)? 1: inner_locc;
					while(inner_carry !=0 && inner_locc<xdim-1){
						eps_px_cnt[inner_locc] += 1;
						eps_px_cnt[inner_locc]%= detail_len;
						inner_carry = (eps_px_cnt[inner_locc]==0)? 1: 0;
						inner_locc += (eps_px_cnt[inner_locc]==0)? 1: 0;
					}
				}// while innerlocc
			}else{
				// not valid, do nothing
			} // if valid
			
			// update the carry 
			eps_pycx_cnt[0] += 1;
			eps_pycx_cnt[0]%= detail_len;
			carry = (eps_pycx_cnt[0]==0)? 1 : 0;
			locc = (eps_pycx_cnt[0]==0)? 1 : locc;
			while(carry!=0 && locc<xdim){
				eps_pycx_cnt[locc]+=1;
				eps_pycx_cnt[locc]%=detail_len;
				carry = (eps_pycx_cnt[locc]==0)? 1 : 0;
				locc += (eps_pycx_cnt[locc]==0)? 1 : 0;
			}
		} // while(locc<4)
		
		// if no two way grid search
		//Mat worst_pxy = (crude_eps_pycx * crude_eps_px.asDiagonal()).transpose();
		//Vec worst_py = worst_pxy.colwise().sum();
		// after two-way grid search, print the result
		cout<<std::setw(5)<<beta<<","\
		<<std::setw(12)<<best_mi<<","\
		<<std::setw(12)<<worst_mi_eps<<","\
		<<std::setw(12)<<worst_err<<","\
		<<std::setw(12)<<worst_kl_train<<","\
		<<std::setw(12)<<worst_kl_x<<","\
		<<std::setw(12)<<worst_kl_y<<",";
		for(size_t tt=0;tt<xdim;++tt){
			cout<<std::setw(12)<<worst_kl_ycx(tt);
			if(tt!=xdim-1)
				cout<<",";
		}
		cout<<endl;
		// write the matrices
		fid<<"[beta]"<<endl<<beta<<endl;
		fid<<"[best_pycx]"<<endl<<best_pycx<<endl;
		fid<<"[eps_pycx]"<<endl<<worst_eps_pycx<<endl;
		fid<<"[eps_px]"<<endl<<worst_eps_px<<endl;

		beta += beta_inc;
	} // for(double beta=1.0.......)
	
	
	// closing the fstream head
	fid.close();

	return 0;
}
