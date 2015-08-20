#!/usr/bin/python
import speaker.recognition as SR
from features import mfcc
from speaker.silence import remove_silence
import scipy.io.wavfile as wav
from time import localtime, strftime
import sys, getopt

def readSegFile(segfn):
    inFile = open(segfn)
    fileText = inFile.read()
    rows = fileText.split('\n')
    segments = []
    for row in rows:
        cols = row.split(' ')
        segments.append(cols)
    return segments

def myLengend(speakermodel):
    lengend = 'SPK_01|'+ strftime("%Y-%m-%d %H:%M", localtime()) + '|Source_Program=SpeakerID.py ' + \
    speakermodel + '|Source_Person=He Xu\n'
    return lengend

def totime(secs):
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    return h, m, s

def readHeader(infofn):    
    inFile = open(infofn)
    fileText = inFile.read()
    lines = fileText.split('\n')
    header = ''
    for line in lines:
        header = header + line + '\n'
        if line.startswith('LBT'):
            break
    endline = lines[-2]
    return header, endline

def readSegFeat(start_t, end_t, signal, sr):
    try:
        sig = signal[int(sr * start_t) : int(sr * end_t)] 
    except:
        sig = signal[int(sr * start_t) : -1]
    cleansig = remove_silence(sr, sig)
    mfcc_vecs = mfcc(cleansig, sr, numcep = 15) 
    return mfcc_vecs

def main(argv):
    if len(argv) != 4:
        print 'usage: SpeakerID.py <trained_model> <wav_file> <segmentation_file> <metainfo_file>'
        sys.exit(2)
        
    speakermodel = argv[0]
    audiofn = argv[1]
    segfn = argv[2]
    infofn = argv[3]
    


    segments = readSegFile(segfn)
    fn = infofn.split('.')[0]
    outFile = open(fn+'.spk', 'w')
    (header, endline) = readHeader(infofn)
    outFile.write(header)
    outFile.write(myLengend(speakermodel))
    speakerRec = SR.GMMRec.load(speakermodel)
    (sr, signal) = wav.read(audiofn)
    if len(signal.shape) > 1:
            signal = signal[:, 0]    
    for i in xrange(len(segments)-1):
        start_time = float(segments[i][3])
        try:
            end_time = float(segments[i+1][3])
        except:
            end_time = start_time + float(segments[i][-5])
        segfeat = readSegFeat(start_time, end_time, signal, sr)
        (speaker, llhd) = speakerRec.predict(segfeat)
        entry = "%d:%02d:%02d|" % totime(start_time) + \
        "%d:%02d:%02d|" % totime(end_time) + "Person=" + speaker + "|Log Likelihood=" + str(llhd) + "\n"
        outFile.write(entry)
    outFile.write(endline)
    outFile.close()
    
if __name__ == "__main__":
    main(sys.argv[1:])