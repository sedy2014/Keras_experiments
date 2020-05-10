
# modes: 1 : jenkins Cape, 2: CAf as371, 3 CAf vs550
build_mode = 1
if build_mode == 1:
    usernames = ['smahotra','apouya']
    tokens = ['110a917ca15996b0c75702ae40ea33241a,110a917ca15996b0c75702ae40ea33241a']
    #params
    jenkins_url = 'http://sjc1-caf-dev.synaptics.com:8080'
    repo_url = 'http://sjsvn01.synaptics.com//svn//audiofw_dsp//capefw'
    #job_name = 'python_created'
    # build option; trunk / tag
    build_opt = 'trunk'
    build_config = '//evk//nebula//rdk_aurora'
    copy_to_filer = 0
    test_platform = 'No-Test'
    include_sdk = 0
    include_release_notes  = 0
    do_reduced_checkout = 0
    rmt_build_pth = '//ussjf-mcorp.synaptics-inc.local//IoT_MLCOLD//Public//builds//standalone//tst1'

# input path
if build_mode == 1:
    sim_pth = 'D://Sid//python_dev//simulator//cape//simulator_11_18'
    # No processing: AURO, With processing : ALEX
    sim_mode = '\"ALEX\"'
else:
    sim_pth =  'D://Sid//python_dev//simulator//caf'

input_file_path = '\"../../../../input\"'
output_file_path = '\"../../../../output\"'
input_file_name = '\"silence_3ft-30deg_Alexa-English_7.19.0.0_6_channel_MIC.wav\"'
echo_ref_file_name = '\"silence_3ft-30deg_Alexa-English_7.19.0.0_6_channel_ECHO.wav\"'
output_file_name = '\"processed.wav\"'
contcl_pth =  'C:/Program Files/Conexant/Sculptor/tcl/Bin/tclsh85.exe'
# for AS371


# for Vs55