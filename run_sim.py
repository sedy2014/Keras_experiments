
from run_sim_config import CapeFWSVN,Configuration,TestPlatform,jenkins_url,\
    usernames,job_name,build_mode,input_file_path,output_file_path,input_file_name,echo_ref_file_name,output_file_name,\
    rmt_build_pth,sim_pth,contcl_pth,sim_mode
from shutil import copyfile
from  shutil import copytree,rmtree
import fileinput
import os
import jenkins
import time,sys,subprocess
import glob, os.path
import signal
usernm  = os.getlogin()
if usernm in  usernames:
    print('username is valid')
else:
    raise Exception('Invalid credentials')

def read_job_params_from_jenkins(j_ser,job_name):
    a= j_ser.get_job_info(job_name)
    build_num = j_ser.get_job_info(job_name)['lastSuccessfulBuild']['number']
    build_info = j_ser.get_build_info(job_name, build_num)
    build_info_actions = build_info['actions']
    l1 = build_info['actions'][0]['parameters']
    param_read = {}
    for item in l1:
        nm = item['name']
        vl = item['value']
        param_read[nm] = vl
    return param_read

def read_job_params_from_config():
    param_read= {}
    param_read['CapeFWSVN'] = CapeFWSVN
    param_read['Configuration'] = Configuration
    param_read['TestPlatform'] = TestPlatform
    return param_read

def comp_set_params(d1,d2):
    d3 = {}
    for k1, v1 in d1.items():
        if k1 in d2.keys():
            d3[k1] = d1[k1]
        else:
         print('the parameter ' + k1 + ' will be ignored')
    return d3

def run_cmd(args):
    print(args)
    p = subprocess.Popen(args, shell=True, stdout=subprocess.PIPE)
    for line in p.stdout:
        print(line)
    p.wait()
    return p.returncode

if build_mode == 1:
    token ='114fc47adcc83955b3795523e6a4c9b0c6'
    j_ser = jenkins.Jenkins(jenkins_url,usernm,token)
    jobs = j_ser.get_jobs()

    print('****************************')
    print('The jobs on server :'+ jenkins_url    + ' are ')
    for item in jobs:
        print(item['name'])
    print('****************************')

    # read job params from jenkins
    par_jen = read_job_params_from_jenkins(j_ser,job_name)
    # read jobs parameters to be set
    par_cfg = read_job_params_from_config()

    # Get final params to be set, that already exist in the job
    par_set_jen = comp_set_params(par_cfg,par_jen)
    print('****************************')
    jobq = j_ser.build_job( job_name , parameters=par_set_jen ,token = token)

    while True:
        print('Running....')
        if j_ser.get_job_info(job_name)['lastSuccessfulBuild']['number'] == j_ser.get_job_info(job_name)['lastBuild']['number']:
            print("Last ID %s, Current ID %s"  % (j_ser.get_job_info(job_name)['lastSuccessfulBuild']['number'], j_ser.get_job_info(job_name)['lastBuild']['number']))
            break
    time.sleep(3)
    print('Stop....')
    print('***********Console op *****************')
    console_output = j_ser.get_build_console_output(job_name, j_ser.get_job_info(job_name)['lastBuild']['number'])
    print(str(console_output))
    print('****************************')
    par_jen_last_build = read_job_params_from_jenkins(j_ser,job_name)
    if os.path.exists(sim_pth):
        rmtree(sim_pth)
        copytree(rmt_build_pth, sim_pth, symlinks=False, ignore=None)

    # remove uncessary files and copy relevant firmware files from remote build location to this vm
    filelist = glob.glob(os.path.join(sim_pth, "*.zip"))
    for f in filelist:
        os.remove(f)

    for subfold in os.listdir(sim_pth + "//fcp"):
        if subfold!= 'scripts':
            rmtree(sim_pth + "//fcp//" + subfold  )
    if os.path.exists(sim_pth + "//firmware") :
        rmtree(sim_pth + "//firmware")
    if os.path.exists(sim_pth + "//sdk"):
        rmtree(sim_pth + "//sdk")
    for x in glob.glob(os.path.join(sim_pth + "//simulator//firmware" , "*.*")):
        if not x.endswith('.glm'):
            os.remove(x)
    for x in glob.glob(os.path.join(sim_pth + "//simulator//tools" , "*.*")):
        if not (x.endswith('.dll') or x.endswith('.tcl')):
            os.remove(x)
    # read each line of tcl file , overwriting the paths
    fl_tcl = sim_pth + "//simulator//tools//RunSim.tcl"
    for line in fileinput.input(fl_tcl, inplace=True):
        if 'set TCLTOOL "tclkitsh-8.5.9-win32.upx.exe"' in line:
                line = line + "\n" + 'set prefix "aurordk"'
        elif  'set prefix "aurordk' in line:
            line = ''
        elif 'getopt argv -d  sim_mode' in line:
            line = 'getopt argv -d  sim_mode' + " " + sim_mode + "\n"
        elif 'getopt argv -pi input_file_path' in line:
            line = 'getopt argv -pi input_file_path' + " " +  input_file_path + "\n"
        elif 'getopt argv -po output_file_path' in line:
            line =  'getopt argv -po output_file_path' + " " + output_file_path + "\n"
        elif 'getopt argv -i input_file_name' in line:
            line = 'getopt argv -i input_file_name'  + " " + input_file_name +"\n"
        elif 'getopt argv -r echo_ref_file_name' in line:
            line = 'getopt argv -r echo_ref_file_name' + " " + echo_ref_file_name + "\n"
        elif 'getopt argv -o output_file_name' in line:
            line = 'getopt argv -o output_file_name' + " " + output_file_name + "\n"
        #print('{} {}'.format(line))
        sys.stdout.write(line)  # for Python 3

    os.chdir('D:\\Sid\\python_dev\\simulator\\cape\\simulator_11_18\\simulator\\tools')
    print("current dir is : " + os.getcwd())
    cmd = [contcl_pth,'RunSim.tcl']

    # p = subprocess.Popen("C:/Program Files/Conexant/Sculptor/tcl/Bin/tclsh85.exe RunSim.tcl", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # stdout, stderr = p.communicate(timeout=600)
    # if p.returncode != 0:
    #     os.kill(os.getpid(), signal.SIGTERM)
    #     raise Exception('There was an error' % stderr)

    # Run the TCL command
    run_cmd(cmd)
else:
    # Build locally
    ip_pth = input_file_path + '\\'+ input_file_name
    op_pth = output_file_path +  '\\' + output_file_name
    #read top script and write paths to it

print('hi')
