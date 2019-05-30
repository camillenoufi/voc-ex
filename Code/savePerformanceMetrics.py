import numpy as np
import seaborn as sn
import pandas as pd
import csv

def savePerformanceMetrics(correct, total, f1_micro, f1_macro, f1_weighted, precision, recall, cm, num_batches, model_file):

    md = {}

    print('Overall Accuracy of the model on the test inputs: {} %'.format((correct / total) * 100))
    md['Overall Accuracy'] = (correct / total) * 100

    print('Average f1, precision, and recall metrics over {} batches:'.format(num_batches))
    md['Num Batches'] = num_batches

    print('F1 (micro):     {}'.format(f1_micro/num_batches))
    md['F1 (micro)'] = f1_micro/num_batches * 100

    print('F1 (macro):     {}'.format(f1_macro/num_batches))
    md['F1 (macro)'] = f1_macro/num_batches * 100

    print('F1 (weighted):  {}'.format(f1_weighted/num_batches))
    md['F1 (weighted)'] = f1_weighted/num_batches * 100

    print('Precision: {}'.format(precision/num_batches))
    md['Precision'] = precision/num_batches * 100

    print('Recall:    {}'.format(recall/num_batches))
    md['Recall'] = recall/num_batches * 100


    cm = np.array(cm)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Confusion Matrix:    {}'.format(cm))

    #save confusion matrix as .csv
    try:

        cm_file = os.path.join('.', model_file + 'cm.csv')

        np.savetxt(cm_file, cm, delimiter=",")
        df_cm = pd.DataFrame(cm, index = [i for i in lbls],
                          columns = [i for i in lbls])
        export_csv = df_cm.to_csv (cm_file, header=True) #Don't forget to add '.csv' at the end of the path
    except:
        print("I/O error")

    #save metrics as csv
    try:
        metrics_file = os.path.join('.',model_file + 'metrics.csv')
        with open(metrics_file, 'wb') as f:
            for key in md.keys():
                f.write("%s,%s\n"%(key,md[key]))
    except:
        print("I/O error")
