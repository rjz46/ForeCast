import requests
import json
import sys
import os
import time
import nltk 
from nltk.sentiment import SentimentIntensityAnalyzer

global_count = 0
get_toxicity_count = 0 
sequence_number = 1
vader_analyzer = SentimentIntensityAnalyzer()

headers = {
    'Content-Type': 'application/json',
}

params = [
    (('key', 'AIzaSyCNy3RLPbblytD5Uejh4GkiBb3wAgHENbI'),),
    (('key', 'AIzaSyDyRDMXjs3UFWxmsAcyBnkTG5dLgK4Jjzw'),),
    (('key', 'AIzaSyBco105Hk0jBIfJi7PVXbBpQpgzVurweUM'),),
    (('key', 'AIzaSyAWzZLs1xGBWNkEi6iiFBOFF37-qWsOivY'),), 
    (('key', 'AIzaSyASejkIh2dtCXhndoD8Cg63oYvakbIKUtg'), ),
    (('key', 'AIzaSyBWo9h7yt2NiupOWJ58-soBpHoWu4FickI'), ),
    (('key', 'AIzaSyCCuoFqbt2dXKS0-z8-72KKbwj_zLZbtEs'), ),
    (('key', 'AIzaSyC4pTatou4rQKxBAzSgxDRoE_r6nkSdrHI'), ),
    (('key', 'AIzaSyCLkTAqnm_OZl4dxIgVa_qQkmV0dIOg5wI'), ),
    (('key', 'AIzaSyAItWACAVrBssCR6Y47iIj1P2af-ouMC3k'), ),
    (('key', 'AIzaSyByP7AGNtOUZEuyDjWDW9X1A9YN6EJOCZU'), ),
    (('key', 'AIzaSyCjkZkWX1k16XFO2OZxHob0yfacHRVSBBE'), ),
    (('key', 'AIzaSyAkxgrwu39qvhYkp_KbbgYRfjt8CNN0vBE'), ),
    (('key', 'AIzaSyBTk-xFCRde2TnzLxMF9DvZ8Pt8Sta3R98'), ),
    (('key', 'AIzaSyDBas3bRqeVreRUbbzjwjufHcOu1YsRB38'), ),
    (('key', 'AIzaSyBPjzjcHZzKTwrxuTJ90zQ3Bd6DZaG8Od4'), ),
    (('key', 'AIzaSyCMAk7WgQszF9gcyJUowH20cfe9d4lT0pQ'), ),
    (('key', 'AIzaSyA0Q9hnn867iZGljX707fow6wo6YU5g7Fk'), ),
    (('key', 'AIzaSyBTU3hI_z_ois-KkXSIdUyxiHNi4RUKEaQ'), ),
    (('key', 'AIzaSyAT1wO4XdJXg0zjJPpMhilE1S23fIt7x-A'), ),
    (('key', 'AIzaSyBA-wDLTmHid5RannQY0cEWBa0VS8G5CWo'), ) #index 20
]


dict_of_ids = {}

def get_toxicity(line):
    global get_toxicity_count
    tox_api_key = params[get_toxicity_count % 21]

    if len(line) > 0:
        try:
            data = '{comment: {text:"'+line+'"}, languages: ["en"], requestedAttributes: {TOXICITY:{}} }'
            response = requests.post('https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze', headers=headers, params=tox_api_key, data=data)
            j = json.loads(response.text)
            get_toxicity_count +=1
            return j['attributeScores']['TOXICITY']['summaryScore']['value']
        except:
            print("ERROR1!!!!!!!!!!!!!!!!!!!!" + str(get_toxicity_count))
            try:
                time.sleep(2)
                data = '{comment: {text:"'+line+'"}, languages: ["en"], requestedAttributes: {TOXICITY:{}} }'
                response = requests.post('https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze', headers=headers, params=tox_api_key, data=data)
                j = json.loads(response.text)
                
                return j['attributeScores']['TOXICITY']['summaryScore']['value']
            except:
                print("ERROR2!!!!!!!!!!!!!!!!!!!!" + str(get_toxicity_count))

                try:
                    time.sleep(2)
                    data = '{comment: {text:"'+line+'"}, languages: ["en"], requestedAttributes: {TOXICITY:{}} }'
                    response = requests.post('https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze', headers=headers, params=tox_api_key, data=data)
                    j = json.loads(response.text)
                    
                    return j['attributeScores']['TOXICITY']['summaryScore']['value']
                except:
                    print("ERROR3!!!!!!!!! " + str(get_toxicity_count))
                    print(j)
    return 0.0


class Node(object):
    def __init__(self, comment_id=None, score=None, level=0, comment_count=0, body_or_title = ""):
        self.comment_id = comment_id
        self.body_or_title = body_or_title
        self.score    = score
        self.level     = level
        self.children  = [] #list of nodes
        self.count = comment_count

    def __repr__(self):        
        return '\n{indent}Node({comment_id},{comment_count},{score},{children})'.format(
                                         indent = self.level*'\t', 
                                         comment_id = self.comment_id,
                                         comment_count = self.count,
                                         score = self.score,
                                         children = repr(self.children))
    def add_child(self, child):
        self.children.append(child)    

def tree_builder(obj, level=0):
    global global_count
    global_count+=1

    if level == 0:
        body_or_title=(obj['title']).encode('utf-8')
    else:
        body_or_title=(obj['body']).encode('utf-8')

    line = ''
    for a in body_or_title:
            if a=='[':
                f=False
                break
            if a==' ' or (a<='Z' and a>='A') or (a<='z' and a>='a') or (a<='9' and a>='0') or a=='?' or a=='.':
                line +=a


    node = Node(comment_id=obj['id'].encode('utf-8'), score=0.0, level=level, comment_count = global_count, body_or_title = line)

    for child in obj['children']:
        node.add_child(tree_builder(child, level=level+1))
    return node



#def traverse(node, path = [], path_scores = {}):
def traverse(node, path_scores, path = []):

    global sequence_number
    path.append(node)
    if len(node.children) == 0:
        if len(path) > 5:
            #print( str(sequence_number) + ' sequence with >5 comments')
            #print(path_scores)
            sequence_number +=1
            tuple_path = []
            for elem in path:
                    x = path_scores.get(elem.comment_id)
                    if path_scores.get(elem.comment_id) == None:
                        score = get_toxicity(elem.body_or_title)
                        path_scores[elem.comment_id] = score
                        elem.score = score 
                    else:
                        elem.score = x
                    tuple_path.append((elem.comment_id, elem.body_or_title, elem.score))
            output_txt.write("{}\n".format(tuple_path))
        path.pop()
    else:
        for child in node.children:
            traverse(child, path_scores, path)
        path.pop()


script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "path.txt" #path+name of output file name
output_file = os.path.join(script_dir, rel_path)
output_txt = open(output_file,"a+")

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path_json = "path.txt" #path+name of input file name
json_filename = os.path.join(script_dir, rel_path_json)

with open(json_filename) as fp:
    json_num = 0
    for line in fp:
        if json_num > -1:
            line = str(line)
            line = json.loads(line)
            tree = tree_builder(line)
            traverse(tree, {})

        #print('this is json num in jsonlist: ' + str(json_num))
        json_num += 1


#print("api calls count: " + str(get_toxicity_count))






