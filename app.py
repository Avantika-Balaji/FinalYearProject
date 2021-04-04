from flask import Flask,render_template,request,session,abort,flash,redirect,Response
import os
import uuid
from tinydb import TinyDB, Query,where
from datetime import date
import ffmpeg
import numpy as np
import tensorflow as tf
from sbd import Params, sbd
from transnet_utils import draw_video_with_predictions, scenes_from_predictions
import shutil
import sys
import cv2
params = Params()
params.CHECKPOINT_PATH = "./model/transnet_model-F16_L3_S2_D256"
net = sbd(params)
app = Flask(__name__)

#client_db = TinyDB('./databases/Clients.json')
#lawyer_db = TinyDB('./databases/Lawyers.json')
m = 'CL-0'
n = 'LA-0'

def sbd(path):
    video_stream, err = (
    ffmpeg.input(path).output('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(params.INPUT_WIDTH, params.INPUT_HEIGHT)).run(capture_stdout=True))
    video = np.frombuffer(video_stream, np.uint8).reshape([-1, params.INPUT_HEIGHT, params.INPUT_WIDTH, 3])
    predictions = net.predict_video(video)
    scenes = scenes_from_predictions(predictions, threshold=0.1)
    # For ilustration purposes, only the visualized scenes are shown.
    print(scenes[:])
    print(scenes[:]/25)
    return scenes

def FrameCapture(path): 
    

    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
    success, image = vidObj.read() 
    # Used as counter variable 
    y=0
    count = 0
    # checks whether frames were extracted 
    success = 1
    while success: 
        
        # vidObj object calls read 
        # function extract frames 
        
  
        # Saves the frames with frame-count 
        if(count%10==0):
            cv2.imwrite("C:/Users/jackg/Desktop/test/%d.jpg" % count, image)  #change it to the correct location
        count+=1

        success, image = vidObj.read() 


def caption_sentence_transform(lines):
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    lines1=lines
    m=0
    while(m<2):
        j=0
        i=0
        n=len(lines1)
        while(j<n-1):
            i=j+1
            while(i<n):
                sentence1=lines1[j]
                sentence2=lines1[i]
                embeddings1 = model.encode(sentence1, convert_to_tensor=True)
                embeddings2 = model.encode(sentence2, convert_to_tensor=True)
                cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
                print(sentence1,"\t",sentence2," ",cosine_scores[0][0])
                #print(cosine_scores[0][0])
                if(cosine_scores[0][0]>0.7):
                    lines1.remove(lines1[i])
                    i-=1
                n=len(lines1)
                i+=1
            j+=1
        m+=1
    return lines1


def captioning_shit(scenes, path):
    

    test_path = "C:/Users/jackg/Desktop/test/"   #change it to the correct location
    if(os.path.isdir(test_path)):
    	shutil.rmtree(test_path)
    	os.mkdir(test_path)
    else:
    	os.mkdir(test_path)

    FrameCapture(path)

    for i in range(len(scenes)):
        os.mkdir(os.path.join(test_path, str(i)))

    for i in range(len(scenes)):
        a,b = scenes[i]
        a=((a//10)+1)*10
        b=((b//10))*10
        print(a,b)
        for j in range(a,b+1,10):
            shutil.move(os.path.join(test_path, str(j)+'.jpg'), os.path.join(test_path, str(i)))

    img_dir_path = "C:/Users/jackg/Desktop/test/"                           #change it to the correct location
    sys.path.insert(1, 'C:/Users/jackg/Desktop/Image-caption/code/')        #change it to the correct location
    from CaptionGenerator import CaptionGenerator 

    caption_generator=CaptionGenerator(
    rnn_model_place='C:/Users/jackg/Desktop/Image-caption/data/caption_en_model40.model',   #change it to the correct location
    dictonary_place='C:/Users/jackg/Desktop/Image-caption/data/MSCOCO/mscoco_caption_train2014_processed_dic.json', 

    #rnn_model_place='C:/Users/avant/Desktop/chainer-caption/data/caption_model40_words.model',  
    #dictonary_place='C:/Users/avant/Desktop/chainer-caption/data/CCTV/cctv_caption_train2014_processed_dic.json', 

    cnn_model_place='C:/Users/jackg/Desktop/Image-caption/data/ResNet50.model',             #change it to the correct location
        #change it to the correct location
    beamsize=3,
    depth_limit=50,
    gpu_id=-1,
    first_word='<sos>',
    )

    f=open("captions.txt","r+")
    f.truncate(0)
    f.close()
    f = open("captions.txt", "a")
    captions_list=""
    caption_transform = []
    for i in range(len(scenes)):
        img_path = os.path.join(img_dir_path, str(i))
        for image_path in os.listdir(img_path):
            input_path = os.path.join(img_path, image_path)
            captions = caption_generator.generate(input_path)
            for caption in captions:
                temp1=(" ".join(caption["sentence"]))
                temp2=(caption["log_likelihood"])
            #print(temp1)
            captions_list+=temp1[6:-6]+'\n'
            #caption_transform.append(temp1[6:-66])
        generated_captions = captions_list.split('\n')
        print("**")
        print(generated_captions)
        print("**")
        temp2 = caption_sentence_transform(generated_captions)

        for cap in temp2:
            #f.write(str(scenes[i])+'\n')
            f.write(cap)
            f.write('=')
        captions_list = ""
        #caption_transform.clear()
        f.write('*')
    #f.write(captions_list)
    f.close()



def get_max(x):
    global m
    m = max(m,x)
    return False

def get_max1(x):
    global n
    n = max(n,x)
    return False
    
def get_id():
    client_db.search(where('uid').test(get_max))
    return 'CL-'+str(int(m.split('-')[1])+1)

def get_id1():
    lawyer_db.search(where('uid').test(get_max1))
    return 'LA-'+str(int(n.split('-')[1])+1)


@app.route('/',methods=['GET'])
def handle_home():
    # session['logged_in'] = True
    # session['user_data'] = {'username':'Brr'}
    # if not session.get('logged_in',False):
    #     return render_template('login.html')
    
    return render_template('mainpage.html')

@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        #uploaded_file.save(uploaded_file.filename)
        path="C:/Users/jackg/Desktop/SBD/testvids/"+uploaded_file.filename  #change it to the correct location
        print(path)
        s=sbd(path)
        captioning_shit(s, path)
        f=open("captions.txt","r")
        a=f.read()
        cap=a.split("==*")
        temp5 = []
        vid_path="C:/Users/jackg/Desktop/SBD/static/"
        final_path=vid_path+uploaded_file.filename
        for caps in cap:
            temp=caps.split('=')
            temp5.append(temp)

        return render_template('mainpage.html',scenes=s, captions=temp5, vid=uploaded_file.filename)



@app.route('/login',methods=['POST'])
def handle_login():
    ''' Check User with database '''
    data = request.form
    if data['password'] == '123' and data['username'] == 'admin@123':
        session['logged_in'] = True
        session['user_data'] = {'username':data['username']}
    else:
        flash('Incorrect Username or Password')
    return redirect('/')

@app.route('/register',methods=['GET','POST'])
def handle_register():
    if request.method == 'GET':
        return render_template('register.html')
    elif request.method == 'POST':
        ''' Check User with database '''
        return redirect('/')

@app.route('/logout',methods=['GET'])
def handle_logout():
    session['logged_in'] = False
    return redirect('/')

@app.route('/clients',methods=['GET','POST'])
def handle_client():
    if not session.get('logged_in',False):
        return render_template('login.html')
    if request.method == 'POST':
        form = request.form
        id = get_id()
        path = './static/files/clients/'+id
        os.mkdir(path)
        os.chmod(path, 777, dir_fd=None, follow_symlinks=True)

        client = {
            'uid': id,
            'First Name': form['fname'],
            'Last Name': form['lname'],
            'Date Of Birth' : form['dob'],
            'Phone' : form['phone'],
            'Start Date' : form['start_date'],
            'Files' : []
        }
        for files in request.files.values():
            files.save(path+'/'+files.filename)
            client['Files'].append(path+'/'+files.filename)
        client_db.insert(client)
        # clients = client_db.all()
 
        return redirect('/clients')
    else:
        clients = client_db.all()
        for x in clients:
            year,month,day = map(int,x['Date Of Birth'].split('-'))
            x['age'] = (int((date.today() - date(year,month,day)).days / 365.2425 ) )
        return render_template('add_client.html',user_data = session['user_data'],clients=clients)


@app.route('/lawyers',methods=['GET','POST'])
def handle_lawyer():
    if not session.get('logged_in',False):
        return render_template('login.html')
    if request.method == 'POST':
        form = request.form
        id = get_id()
        path = './static/files/lawyers/'+id
        os.mkdir(path)
        os.chmod(path, 777, dir_fd=None, follow_symlinks=True)

        lawyer = {
            'uid': id,
            'First Name': form['fname'],
            'Last Name': form['lname'],
            'Date Of Birth' : form['dob'],
            'Phone' : form['phone'],
            'Start Date' : form['start_date'],
            'Files' : [],
            'Area of Expertise' : form['aoe']
        }
        for files in request.files.values():
            files.save(path+'/'+files.filename)
            lawyer['Files'].append(path+'/'+files.filename)
        lawyer_db.insert(lawyer)
        # lawyers = lawyer_db.all()
 
        return redirect('/lawyers')
    else:
        lawyers = lawyer_db.all()
        for x in lawyers:
            year,month,day = map(int,x['Date Of Birth'].split('-'))
            x['age'] = (int((date.today() - date(year,month,day)).days / 365.2425 ) )
        return render_template('add_lawyer.html',user_data = session['user_data'],lawyers=lawyers)


@app.route('/testing1')
def row_click():
    return 'Hello World!'

@app.route('/clients/<id>',methods=['GET'])
def handle_get_client(id):
    if not session.get('logged_in',False):
        return render_template('login.html')
    u = Query()
    res = client_db.search(u.uid == id)
    for i in range(len(res[0]['Files'])):
        if 'summary' in res[0]['Files'][i]:
            res[0]['Files'][i] = 'summary\n'+open(res[0]['Files'][i],'r').read()

    #print(res[0]['Files'])
    # try:
    return render_template('client.html',client_data=res[0],user_data=session['user_data'])
    # except:
    #     return 'No'

    

# @app.errorhandler(404) 
# def invalid_route(e): 
#     return render_template('404.html')




if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(port=3000)
