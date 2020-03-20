from preprocessConversations1 import *
import torch.nn as nn
import torch.optim as optim
import math
import random
import os
import pickle
import time
from torch.nn import init
from tqdm import tqdm
from torch.autograd import Variable
import  matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.input_dim = 500
        self.hidden_dim = 500
        #self.output_dim = 500
        self.num_rnn_layers = 1
        self.nonlinearity = 'tanh'
        self.sigmoid = nn.Sigmoid()

        self.log_softmax = nn.LogSoftmax()
        self.loss = nn.NLLLoss()
        self.rnn = nn.GRU(input_size = self.input_dim, hidden_size = self.hidden_dim, num_layers = self.num_rnn_layers, batch_first=True, bidirectional=True)
        #self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def get_loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)
    
    def forward(self, inputs):
        h0 = Variable(torch.zeros(self.num_rnn_layers*2, inputs.size(0), self.hidden_dim))
        out, hn = self.rnn(inputs, h0)
        # z1 = self.fc(hn[:, -1, :])
        hn = hn[-2, :, :] + hn[-1, : ,:]
        # a1 = self.sigmoid(z1)
        # # print(z1)
        # print(a1[0])
        return hn

class contextRNN(nn.Module):
    def __init__(self):
        super(contextRNN, self).__init__()
        self.input_dim = 500
        self.hidden_dim = 256
        self.output_dim = 1
        self.num_rnn_layers = 1
        self.nonlinearity = 'tanh'
        self.sigmoid = nn.Sigmoid()

        self.log_softmax = nn.LogSoftmax()
        self.loss = nn.NLLLoss()
        #self.rnn = nn.RNN(input_size = self.input_dim, hidden_size = self.hidden_dim, num_layers = self.num_rnn_layers, batch_first=True, nonlinearity=self.nonlinearity)
        self.rnn = nn.GRU(input_size = self.input_dim, hidden_size = self.hidden_dim, num_layers = self.num_rnn_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def get_loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)
    
    def forward(self, inputs):
        h0 = Variable(torch.zeros(self.num_rnn_layers*2, inputs.size(0), self.hidden_dim))
        out, hn = self.rnn(inputs, h0)
        z1 = self.fc((out[:, -1, :self.hidden_dim] + out[:,0,self.hidden_dim:])/2)
        #z1 = self.fc(hn[:, -1, :])
        a1 = self.sigmoid(z1)
        # # print(z1)
        # print(a1[0])
        return a1

def performTrain(model, optimizer, train_data, train_label):
    c = list(zip(train_data, train_label))
    random.shuffle(c)
    train_data, train_label = zip(*c)

    N = len(train_data)
    correct = 0
    total = 0
    totalloss = 0
    minibatch_size = 16
    criterion = nn.BCEWithLogitsLoss()

    for minibatch_index in tqdm(range(N // minibatch_size)):
        optimizer.zero_grad()
        loss = None
        for example_index in range(minibatch_size):
            input_vector = train_data[minibatch_index * minibatch_size + example_index]
            gold_label = train_label[minibatch_index * minibatch_size + example_index]
            predicted_vector = model(input_vector.float())
            #print(predicted_vector)
            if predicted_vector > 0.5:
                predicted_label = 1
            else:
                predicted_label = 0
            correct += int(predicted_label == gold_label)
            total +=1
            predicted_vector = predicted_vector.squeeze(1)
            predicted_vector = predicted_vector.squeeze(0)
            l = criterion(predicted_vector, torch.tensor(gold_label))
            if(loss is None):
                loss = l
            else:
                loss += l
            
        loss = loss / minibatch_size
        loss.backward(retain_graph=True)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        totalloss += loss
    accuracy = (correct / total) * 100
    return totalloss/(N // minibatch_size), accuracy

def validate(model, val_data, val_label):
    correct = 0
    total = 0
    true_label = []
    pred_label = []
    loss = None
    criterion = nn.BCEWithLogitsLoss()

    for i in tqdm(range(len(val_data))):
        input_vector = val_data[i]
        gold_label = val_label[i]
        true_label.append(gold_label)

        predicted_vector = model(input_vector.float())

        if predicted_vector > 0.5:
            predicted_label = 1
        else:
            predicted_label = 0
        pred_label.append(predicted_label)

        correct += int(predicted_label == gold_label)
        total +=1
        
        predicted_vector = predicted_vector.squeeze(1)
        predicted_vector = predicted_vector.squeeze(0)
        l = criterion(predicted_vector, torch.tensor(gold_label))

        if(loss is None):
            loss = l
        else:
            loss += l
    loss = loss / len(val_data)
    accuracy = (correct / len(val_data)) * 100
    fscore  = f1_score(true_label,pred_label)
    recall = recall_score(true_label,pred_label)
    precision = precision_score(true_label,pred_label)

    #precision1, recall1, fscore1, support1 = score(true_label, pred_label)

    target_names = ['class 0', 'class 1']
    print(classification_report(true_label, pred_label, target_names=target_names))
    print(pred_label)
    return loss.data, accuracy, fscore, recall, precision

def main():
    output_rnn1 = []
    data, labels = readData("politics1_nov20.txt")
    # model = RNN()
    # model = model.float()
    # optimizer = optim.Adagrad(model.parameters(),lr=0.001)
    # criterion = nn.BCEWithLogitsLoss()
    # for seq in data:
    #     rnn_sequence = []
    #     for comment in seq:
    #         predicted_vector_comment = model(comment.float())
    #         rnn_sequence.append(predicted_vector_comment)
    #     output_rnn1.append(rnn_sequence)
    
    #Training the context RNN
    train_accuracy = []
    train_losses = []
    val_accuracy = []
    val_fscore = []
    val_recall = []
    val_precision = []
    val_losses = []
    #train_data, val_data, train_label, val_label = getTrainingAndValData(output_rnn1, labels)
    train_data, val_data, train_label, val_label = getTrainingAndValData(data, labels)
    
    num_epochs = 20
    contextmodel = contextRNN()
    contextmodel = contextmodel.float()
    contextoptimizer = optim.Adagrad(contextmodel.parameters(),lr=0.009)
    contextmodel.load_state_dict(torch.load("GRU/GRUmodel1.pth"))

    for e in range(num_epochs):
        loss, accuracy = performTrain(contextmodel, contextoptimizer, train_data, train_label)
        print("Training accuracy for epoch {}: {}".format(e + 1, accuracy))
        train_losses.append(loss)
        train_accuracy.append(accuracy)

        vloss, val_acc, fscore, recall, precision = validate(contextmodel, val_data, val_label)
        print("Validation accuracy for epoch {}: {}".format(e + 1, val_acc))
        print(fscore, recall, precision)
        val_losses.append(vloss)
        val_accuracy.append(val_acc)
        val_fscore.append(fscore)
        val_precision.append(precision)
        val_recall.append(recall)

        #saving model aftr every epoch
        path = "GRU/GRUmodel"
        torch.save(contextmodel.state_dict(),path + str(e+2) + ".pth")

    print("Training Set Metrics")
    print(train_accuracy)
    print(train_losses)

    print("Validation Set Metrics")
    print(val_losses)
    print(val_accuracy)
    print(val_fscore)
    print(val_precision)
    print(val_recall)

    print("Number of Parameters")
    # Number of parameters
    pytorch_total_params = sum(p.numel() for p in contextmodel.parameters() if p.requires_grad)
    for p in contextmodel.parameters():
        if p.requires_grad:
            print(p.numel())
    print(pytorch_total_params)

    # training loss 
    iteration_list = [i+1 for i in range(num_epochs)]
    plt.plot(iteration_list,train_losses)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Training Loss")
    plt.title("RNN: Loss vs Number of Epochs")
    #plt.show()
    plt.savefig('train_loss_history.png')
    plt.clf()
    
    # training accuracy
    plt.plot(iteration_list,train_accuracy)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Training Accuracy")
    plt.title("RNN: Accuracy vs Number of Epochs")
    #plt.show()
    plt.savefig('train_accuracy_history.png')
    plt.clf()

    # validation loss 
    plt.plot(iteration_list,val_losses,color = "red")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Validation Loss")
    plt.title("RNN: Loss vs Number of Epochs")
    #plt.show()
    plt.savefig('val_loss_history.png')
    plt.clf()

    # validation accuracy
    plt.plot(iteration_list,val_accuracy,color = "red")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Validation Accuracy")
    plt.title("RNN: Accuracy vs Number of Epochs")
    #plt.show()
    plt.savefig('val_accuracy_history.png')
    plt.clf()

    # validation fscore
    plt.plot(iteration_list,val_fscore,color = "red")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Validation FScore")
    plt.title("RNN: Fscore vs Number of Epochs")
    #plt.show()
    plt.savefig('val_fscore_history.png')
    plt.clf()

    # validation recall
    plt.plot(iteration_list,val_recall,color = "red")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Validation Recall")
    plt.title("RNN: Recall vs Number of Epochs")
    #plt.show()
    plt.savefig('val_recall_history.png')
    plt.clf()

    # validation precision
    plt.plot(iteration_list,val_precision,color = "red")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Validation Precision")
    plt.title("RNN: Precision vs Number of Epochs")
    #plt.show()
    plt.savefig('val_precision_history.png')
    plt.clf()

main()