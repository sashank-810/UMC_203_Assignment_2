# Question 1

import Oracle_Assignment_2 as oa
import requests
import numpy as np
import matplotlib.pyplot as plt
data = oa.q1_get_cifar100_train_test(23634)

training_data = data[0]
test_data = data[1]
X_train = []
y_train = []

for i in training_data :
    X_train.append(i[0])
    y_train.append(i[1])
print(len(y_train))
X_test = []
y_test = []
for i in test_data :
    X_test.append(i[0])
    y_test.append(i[1])
print(len(y_test))
print(len(X_train))

print(len(X_train))
print(len(y_train))

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

wprime = np.zeros(X_train.shape[1])
print(wprime)
bprime = 0


def perceptron(X, y, max_iter, w, b) :
    miss_rate = []
    for i in range(max_iter) :
        for j in range(X.shape[0]) :
            if y[j] * (np.dot(w, X[j]) + b) <= 0 :
                w = w + y[j] * X[j]
                b += y[j]
        miss = 0
        for j in range(X.shape[0]) :
            if y[j] * (np.dot(w, X[j])+b) <= 0 :
                miss += 1
        miss_rate.append(miss/X.shape[0])
        if miss == 0 :
            print(f"The algorithm has converged at {i}th iteration")
            break
        if i == max_iter-1 and miss != 0 :
            print("The algorithm did not converge")
    # print(len(miss_rate))
    return w, b, miss_rate



for i in [100] :
    wprime = np.zeros(X_train.shape[1])
    bprime = 0
    wprime, bprime, miss_rateprime = perceptron(X_train, y_train, i, wprime, bprime)
    plt.plot(range(1, len(miss_rateprime) + 1), miss_rateprime)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Misclassification Rate")
    plt.title("Misclassification Rate vs. Iterations")
    plt.grid()
    plt.show()




count = 0
for i in range(X_test.shape[0]) :
    if y_test[i] * (np.dot(wprime, X_test[i])+bprime) <= 0 :
        count += 1
print(count)
print(count/200)

import cvxopt
import time


# +
# SVM construction using cvxopt
class PrimalSVM :
    def __init__(self, C = 1.0) :
        self.w = None
        self.b = None
        self.C = C
    def fit(self, X, y) :
        def minimise(X, y) :
            ns, nf = X.shape
            P = np.zeros((nf+1+ns, nf+1+ns))
            for i in range(nf) :
                P[i, i] = 1
            q = np.zeros(nf+1+ns)
            q[nf + 1:] = self.C * np.ones(ns)

            G = np.zeros((2*ns, nf+1+ns))
            h = np.zeros(2*ns)
            # Implementing first constraint :- yi(w.xi + b) >= 1-epsiloni
            for i in range(ns):
                G[i, :nf] = -y[i] * X[i]  
                G[i, nf] = -y[i]
                G[i, nf + 1 + i] = -1
                h[i] = -1
            # Implementing second constraint :- epsilon >= 0
            for i in range(ns):
                G[ns + i, nf + 1 + i] = -1
                h[ns + i] = 0  

            P = cvxopt.matrix(P)
            q = cvxopt.matrix(q)
            G = cvxopt.matrix(G)
            h = cvxopt.matrix(h)
    # Forgot that we should be calculating time taken also
            cvxopt.solvers.options['show_progress'] = False
            start = time.time()
            sol = cvxopt.solvers.qp(P, q, G, h)
            end = time.time()
            print("Time taken Primal", end-start)
            return sol
        sol = minimise(X, y)
        alpha = np.array(sol['x']).flatten()

        self.w = alpha[:X.shape[1]]
        self.b = alpha[X.shape[1]]
        slack_vars = alpha[X.shape[1] + 1:]
        return self.w, self.b, slack_vars

svm = PrimalSVM()

wprimal,bprimal,slack_values = svm.fit(X_train, y_train)
# print("Weight vector (w):", svm.w)
# print("Bias term (b):", svm.b)
print("Slack variables:", slack_values)
print(len(slack_values))
print(f"Number of  epsilon > 1: {np.sum(slack_values > 1)}")
extractor = []
for i in range(len(slack_values)) :
    if slack_values[i] > 1 :
        extractor.append(i)
print(extractor)
# Add extractor to csv file
import csv

with open('inseparable_23634.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # each element should be in a separate line
    for i in extractor:
        writer.writerow([i])
# -

X_train = list(X_train)
y_train = list(y_train)
X_trainprimal = [element for i, element in enumerate(X_train) if i not in extractor]
y_trainprimal = [element for i, element in enumerate(y_train) if i not in extractor]
print(len(X_trainprimal))
print(len(y_trainprimal))
X_trainprimal = np.array(X_trainprimal)
y_trainprimal = np.array(y_trainprimal)
print(X_trainprimal.shape)

wnew = np.zeros(X_trainprimal.shape[1])
bnew = 0
miss_rate_new = []
wnew, bnew, miss_rate_new = perceptron(X_trainprimal, y_trainprimal, 500000, wnew, bnew)
print(len(miss_rate_new))
plt.plot(range(1, len(miss_rate_new) + 1), miss_rate_new)
plt.xlabel("Number of Iterations")
plt.ylabel("Misclassification Rate")
plt.title("Misclassification Rate vs. Iterations")
plt.grid()
plt.show()


# +
X_train = np.array(X_train)
y_train = np.array(y_train)
class DualSVM :
    def __init__(self, C = 1.0) :
        self.C = C
        self.alpha = None
        self.b = None
        self.w = None
    def fit(self, X, y) :
        ns, nf = X.shape
        K = np.zeros((ns, ns))
        for i in range(ns) :
            for j in range(ns) :
                K[i, j] = np.dot(X[i], X[j])
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-np.ones(ns))
        G = cvxopt.matrix(np.vstack((-np.eye(ns), np.eye(ns))))
        h = cvxopt.matrix(np.hstack((np.zeros(ns), self.C * np.ones(ns))))
        A = cvxopt.matrix(y, (1, ns))
        b = cvxopt.matrix(0.0)
        cvxopt.solvers.options['show_progress'] = False
        start = time.time()
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        end = time.time()
        print("Time taken Dual", end-start)
        alpha = np.array(sol['x']).flatten()
        self.alpha = alpha
        self.w = np.sum((alpha * y).reshape(-1, 1) * X, axis=0)
        sv_indices = (alpha > 1e-5) & (alpha < self.C - 1e-5)
        if np.any(sv_indices):
            self.b = np.mean(y[sv_indices] - np.dot(X[sv_indices], self.w))
        else:
            self.b = 0

        return self.alpha, self.w, self.b
    
dualsvm = DualSVM()
alpha, wdual, bdual = dualsvm.fit(X_train, y_train)
print("Weight vector (w):", wdual)
print("Bias term (b):", bdual)
print("Alpha values:", dualsvm.alpha)
print(len(dualsvm.alpha))
# -

X_traindual = []
y_traindual = []
gay = []
for i in range(X_train.shape[0]) :
    if y_train[i] * (np.dot(wdual, X_train[i])+bdual) <= 0 :
        gay.append(i)
    else :
        X_traindual.append(X_train[i])
        y_traindual.append(y_train[i])
X_traindual = np.array(X_traindual)
y_traindual = np.array(y_traindual)
print(X_traindual.shape)
print(gay == extractor)


# +
class DualSVMrbf:
    def __init__(self, C=10.0):
        self.C = C
        self.alpha = None
        self.b = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_alphas = None
        self.gamma = None

    def fit(self, X, y, gamma):
        self.gamma = gamma
        ns, nf = X.shape
        K = np.zeros((ns, ns))

        # Compute the RBF kernel matrix
        for i in range(ns):
            for j in range(ns):
                K[i, j] = np.exp(-gamma * np.linalg.norm(X[i] - X[j]) ** 2)

        # Define parameters for quadratic programming
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-np.ones(ns))
        G = cvxopt.matrix(np.vstack((-np.eye(ns), np.eye(ns))))
        h = cvxopt.matrix(np.hstack((np.zeros(ns), self.C * np.ones(ns))))
        A = cvxopt.matrix(y.reshape(1, -1), (1, ns), "d")
        b = cvxopt.matrix(0.0)

        # Solve the quadratic programming problem
        cvxopt.solvers.options['show_progress'] = False
        start = time.time()
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        end = time.time()
        print("Time taken for Dual Optimization:", end - start)
        print(self.gamma)

        # Extract Lagrange multipliers (α)
        alpha = np.array(sol['x']).flatten()
        self.alpha = alpha

        # Select support vectors where 0 < α < C
        sv_indices = np.where(np.logical_and(alpha > 1e-5, alpha < self.C - 1e-5))[0]
        
        if len(sv_indices) == 0:
            print("No valid support vectors found.")
            self.b = 0
            return self.alpha, self.b

        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]
        self.support_vector_alphas = alpha[sv_indices]

        # Compute bias term `b` using only support vectors
        b_values = []
        for i in range(len(sv_indices)):
            b_i = self.support_vector_labels[i] - np.sum(
                self.support_vector_alphas * self.support_vector_labels * 
                np.exp(-gamma * np.linalg.norm(self.support_vectors - self.support_vectors[i], axis=1) ** 2)
            )
            b_values.append(b_i)

        self.b = np.mean(b_values)

        return self.alpha, self.b

    def predict(self, X):
        if self.support_vectors is None or self.support_vector_alphas is None:
            raise ValueError("The model is not trained yet. Call `fit` first.")

        predictions = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            kernel_values = np.exp(-self.gamma * np.linalg.norm(self.support_vectors - X[i], axis=1) ** 2)
            predictions[i] = np.sum(self.support_vector_alphas * self.support_vector_labels * kernel_values) + self.b
        
        return np.sign(predictions)

    
dualsvmrbf = DualSVMrbf()
gammas = [0.01, 0.1, 1, 10, 20,25,30,50,75,100]
for gamma in gammas :
    alpha, bdualrbf = dualsvmrbf.fit(X_train, y_train, gamma)
    y_pred = dualsvmrbf.predict(X_train)
    countrbf = 0
    for i in range(y_pred.shape[0]) :
        if y_pred[i] != y_train[i] :
            countrbf += 1
    print(countrbf)
# print("Bias term (b):", bdualrbf)
# print("Alpha values:", dualsvmrbf.alpha)


# Question 2 



import Oracle_Assignment_2 as oa
import requests
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
oa.q2_get_mnist_jpg_subset(23629)


from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

# +

dataset = datasets.ImageFolder(root="q2_data", transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



# +
def evaluate(model, train_loader, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")
    correct, total = 0, 0
    with torch.no_grad() :
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Train Accuracy: {100 * correct / total:.2f}%")


def train(model, train_loader, criterion, optimizer, device, epochs, test_loader):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        evaluate(model, train_loader, test_loader, device)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")


def metricslearn(model, train_loader, test_loader, device):
    model.eval()
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.tolist())
            y_true.extend(labels.tolist())
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


# +
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

modelMLP = MLP().to("cpu")
criter = nn.CrossEntropyLoss()
optimi = optim.SGD(modelMLP.parameters(), lr=0.01, momentum=0.9)
train(modelMLP, train_loader, criter, optimi, "cpu", epochs=17, test_loader = test_loader)
evaluate(modelMLP, train_loader, test_loader, "cpu")


# -

print("Confusion Matrix of MLP")
metricslearn(modelMLP, train_loader, test_loader, "cpu")


# +
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


modelCNN = CNN().to("cpu")
criter = nn.CrossEntropyLoss()
optimi = optim.SGD(modelCNN.parameters(), lr=0.01, momentum=0.9)

train(modelCNN, train_loader, criter, optimi, "cpu", epochs=17, test_loader=test_loader)
evaluate(modelCNN, train_loader, test_loader, "cpu")
metricslearn(modelCNN, train_loader, test_loader, "cpu")


# +
def dataloader_to_numpy(dataloader):
    data_list = []
    labels_list = []

    for images, labels in dataloader:
        images = images.numpy().reshape(images.shape[0], -1)
        data_list.append(images)
        labels_list.append(labels.numpy())

    data_array = np.vstack(data_list)
    labels_array = np.concatenate(labels_list)
    return data_array, labels_array

train_data, train_labels = dataloader_to_numpy(train_loader)
test_data, test_labels = dataloader_to_numpy(test_loader)

# -

def pca(data, k) :
    cov_matrix = np.cov(data.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = eigenvalues[idx]
    eigenvecks = eigenvectors[:, :k]
    new_data = np.dot(data, eigenvecks)
    return new_data, eigenvecks


mean = np.mean(train_data, axis=0)
std = np.std(train_data, axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std
k = 117
train_data_pca, eigenvecks = pca(train_data, k)
test_data_pca = np.dot(test_data, eigenvecks)


# +
train_data_pca_tensor = torch.tensor(train_data_pca, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
test_data_pca_tensor = torch.tensor(test_data_pca, dtype=torch.float32)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

train_pca_dataset = TensorDataset(train_data_pca_tensor, train_labels_tensor)
test_pca_dataset = TensorDataset(test_data_pca_tensor, test_labels_tensor)

train_pca_loader = DataLoader(train_pca_dataset, batch_size=32, shuffle=True)
test_pca_loader = DataLoader(test_pca_dataset, batch_size=32, shuffle=False)

sample = train_data_pca[29]
label = train_labels[29]
reconstructed = np.dot(sample, eigenvecks.T)
reconstructed = (reconstructed * std) + mean
reconstructed = reconstructed.reshape(28, 28)
plt.imshow(reconstructed, cmap="gray")
plt.title(f"Label: {label}")


# +
class MLPPCA(nn.Module):
    def __init__(self):
        super(MLPPCA, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(117, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
modelMLPPCA = MLPPCA()


modelMLPPCA = MLPPCA().to("cpu")
criter = nn.CrossEntropyLoss()
optimi = optim.SGD(modelMLPPCA.parameters(), lr=0.01, momentum=0.9)
train(modelMLPPCA, train_pca_loader, criter, optimi, "cpu", epochs=17, test_loader = test_pca_loader)
metricslearn(modelMLPPCA, train_pca_loader, test_pca_loader, "cpu")


# +

class Log_MC_Reg(nn.Module):
    def __init__(self, input_dim=784, output_dim=10):
        super(Log_MC_Reg, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)

logistic_regression = Log_MC_Reg(input_dim = k)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(logistic_regression.parameters(), lr=0.01, momentum=0.9)

train(logistic_regression, train_pca_loader, criterion, optimizer, "cpu", epochs=17, test_loader=test_pca_loader)
# -

metricslearn(logistic_regression, train_pca_loader, test_pca_loader, "cpu")


# +
class Logistic_Binary_Regression(nn.Module):
    def __init__(self, input_dim=784):
        super(Logistic_Binary_Regression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def train_one_vs_rest_classifiers(train_loader, num_classes, input_dim, lr, epochs):
    classifiers = []
    
    for class_index in range(num_classes):
        print(f"Training classifier for class {class_index}")
        model = Logistic_Binary_Regression(input_dim)
        criterion = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        
        for epoch in range(epochs):
            total_batch_loss = 0
            for features, labels in train_loader:
                labels_ovr = []
                for i in range(len(labels)):
                    if labels[i] == class_index:
                        labels_ovr.append(1)
                    else:
                        labels_ovr.append(0)
                labels_ovr = torch.tensor(labels_ovr, dtype=torch.float32).view(-1, 1)
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels_ovr)
                loss.backward()
                optimizer.step()
                total_batch_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_batch_loss/len(train_loader):.4f}")
        
        classifiers.append(model)
    
    return classifiers

def predict_one_vs_rest_classifiers(classifiers, data_loader):
    predictions = []
    
    for features, _ in data_loader:
        batch_predictions = torch.cat([classifier(features) for classifier in classifiers], dim=1)
        predictions.append(batch_predictions)
    
    return torch.cat(predictions)

ovr_classifiers = train_one_vs_rest_classifiers(train_pca_loader, num_classes=10, input_dim=k, lr=0.01, epochs=17)


# +
def compute_roc_curve(y_true, y_scores, num_thresholds=100):
    thres = np.linspace(0, 1, num_thresholds)
    tpr_values = []
    fpr_values = []
    
    for threshold in thres:
        y_pred = (y_scores >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr_values.append(tpr)
        fpr_values.append(fpr)
    
    return np.array(fpr_values), np.array(tpr_values)

def compute_auc(fpr, tpr):
    return -np.trapezoid(tpr, fpr)

def plot_roc_curve(classifiers, data_loader, num_classes):
    for class_index in range(num_classes):
        plt.figure()
        y_true_list = []
        y_scores_list = []
        for features, labels in data_loader:
            outputs = classifiers[class_index](features).detach().numpy()
            y_scores_list.append(outputs.flatten())
            y_true_list.append((labels.numpy() == class_index).astype(int))
            
        y_true = np.concatenate(y_true_list)
        y_scores = np.concatenate(y_scores_list)
        
        fpr, tpr = compute_roc_curve(y_true, y_scores)
        auc_value = compute_auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'Class {class_index} (AUC = {auc_value:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for Class {class_index}')
        plt.legend()
        plt.show()
    
plot_roc_curve(ovr_classifiers, test_pca_loader, num_classes=10)



# Question 3.1

import Oracle_Assignment_2 as oa
import numpy as np
data1 = oa.q3_linear_1(23634)
data2 = oa.q3_linear_2(23634)
X_train_1 = data1[0]
Y_train_1 = data1[1]
X_test_1 = data1[2]
Y_test_1 = data1[3]
X_train_2 = data2[0]
Y_train_2 = data2[1]
X_test_2 = data2[2]
Y_test_2 = data2[3]


print(len(X_train_1[0]))
X_train_1 = np.array(X_train_1)
print(X_train_1.shape)
Y_train_1 = np.array(Y_train_1)
print(Y_train_1.shape)
X_test_1 = np.array(X_test_1)
print(X_test_1.shape)
Y_test_1 = np.array(Y_test_1)
print(Y_test_1.shape)

print(len(X_train_2[0]))
X_train_2 = np.array(X_train_2)
print(X_train_2.shape)
Y_train_2 = np.array(Y_train_2)
print(Y_train_2.shape)
X_test_2 = np.array(X_test_2)
print(X_test_2.shape)
Y_test_2 = np.array(Y_test_2)
print(Y_test_2.shape)


def wlinreg(X,Y):
    return (np.linalg.pinv(X.T @ X)) @ X.T @ Y


# +
def test(X,Y,W):
    return np.mean((X.dot(W)-Y)**2)

W1lin = wlinreg(X_train_1,Y_train_1)
print("OLS for D1_train", test(X_train_1,Y_train_1,W1lin))
print("OLS for D1_test", test(X_test_1,Y_test_1,W1lin))
print("W1", W1lin)

W2lin = wlinreg(X_train_2,Y_train_2)
np.savetxt("w_ols_23634.csv", W2lin, delimiter = ",")
print("OLS for D2_train",test(X_train_2,Y_train_2,W2lin))
print("OLS for D2_test",test(X_test_2,Y_test_2,W2lin))


# -

def wridge(X,Y,l):
    I = np.eye(len(X[0]))
    return np.linalg.pinv(X.T.dot(X)+l*I).dot(X.T).dot(Y)


# +
W1ridge = wridge(X_train_1,Y_train_1,1)
print("RR for D1_train", test(X_train_1,Y_train_1,W1ridge))
print("RR for D1_test", test(X_test_1,Y_test_1,W1ridge))
print("W1", W1ridge)

W2ridge = wridge(X_train_2,Y_train_2,1)
np.savetxt("w_ridge_23634.csv", W2ridge, delimiter = ",")
print("RR for D2_train", test(X_train_2,Y_train_2,W2ridge))
print("RR for D2_test", test(X_test_2,Y_test_2,W2ridge))



# Question 3.2

import Oracle_Assignment_2 as oa2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mystock = oa2.q3_stocknet(23634)
print(mystock) #BAC

from sklearn.preprocessing import StandardScaler
data = pd.read_csv('BAC.csv')
closing_prices = data['Close'].tolist()
closing_prices = np.array(closing_prices)
closing_prices = closing_prices.reshape(-1,1)
scaler = StandardScaler()
scaler.fit(closing_prices)
closing_prices = scaler.transform(closing_prices)
closing_prices = closing_prices.flatten()
print(closing_prices.shape)
def create_data(closing_prices, t):
    # Obtain the labels y by removing the first t elements of closing_price.
    X = []
    Y = []
    for i in range(len(closing_prices)-t):
        X.append(closing_prices[i:i+t])
        Y.append(closing_prices[i+t])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


import cvxopt
class SlackLinearSVRDual:
    def __init__(self, C=1.0, epsilon=0.1):
        self.C = C
        self.epsilon = epsilon
        self.a = None
        self.a_hat = None
    def fit(self, X, y):
        K = X @ X.T
        ns, nf = X.shape
        P = np.block([[K, -K], [-K, K]])
        P = 0.5 * (P + P.T)
        q = np.hstack([self.epsilon - y, self.epsilon + y])
        A = np.hstack([np.ones(ns), -np.ones(ns)]).reshape(1, -1)
        b = np.array([0.0])
        
        # Fix G and h
        G = np.vstack([np.eye(2 * ns), -np.eye(2 * ns)]).astype(np.float64)
        h = np.hstack([self.C * np.ones(2 * ns), np.zeros(2 * ns)]).astype(np.float64)
        
        # Convert to cvxopt matrices
        P = cvxopt.matrix(P, tc='d')
        q = cvxopt.matrix(q, tc='d')
        A = cvxopt.matrix(A, tc='d')
        b = cvxopt.matrix(b, tc='d')
        G = cvxopt.matrix(G, tc='d')
        h = cvxopt.matrix(h, tc='d')
        
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x']).flatten()
        self.a = alphas[:ns]
        self.a_hat = alphas[ns:]
    def get_dual_variables(self):
        return self.a, self.a_hat
svrlin = SlackLinearSVRDual(C=1.0, epsilon=0.1)
def plot_results(Y_test, Y_pred, moving_avg, t):
    plt.figure()
    plt.plot(Y_test, label='Actual Closing Price', color='blue')
    plt.plot(Y_pred, label='Predicted Closing Price', color='red', linestyle='dashed')
    plt.plot(moving_avg, label=f'Average Price', color='black', linestyle='dotted')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title(f'SVR Prediction (t={t})')
    plt.legend()
    plt.show()
for t in [7,30,90] :
    X, Y = create_data(closing_prices, t)
    print(X.shape, Y.shape)
    X_train = X[:int(0.5*len(X))]
    Y_train = Y[:int(0.5*len(Y))]
    X_test = X[int(0.5*len(X)):]
    Y_test = Y[int(0.5*len(Y)):]
    svrlin.fit(X_train, Y_train)
    Y_pred = []
    a, a_hat = svrlin.get_dual_variables()
    w = X_train.T @ (a - a_hat)
    support_vectors = np.where((a - a_hat > 1e-5))[0]
    print('t =', t)
    print('Number of support vectors:', len(support_vectors))
    b = np.mean(Y_train[support_vectors] -(X_train[support_vectors] @ w)) if len(support_vectors) > 0 else 0
    for i in range(len(X_test)):
        Y_pred.append(X_test[i] @ w + b)
    Y_pred = np.array(Y_pred)
    Y_pred = scaler.inverse_transform(Y_pred.reshape(-1,1)).flatten()
    Y_test = scaler.inverse_transform(Y_test.reshape(-1,1)).flatten()
    moving_avg = np.convolve(Y_test.flatten(), np.ones(t)/t, mode='valid')
    plot_results(Y_test, Y_pred, moving_avg, t)


# +

class SlackGaussianSVRDual:
    def __init__(self, C=1.0, epsilon=0.1, gamma=0.1):
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.a = None
        self.a_hat = None
        self.X_train = None
        self.Y_train = None
    
    def gaussian_kernel(self, X1, X2):
        return np.exp(-self.gamma * (np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)))
    
    def fit(self, X, y):
        self.X_train = X
        self.Y_train = y
        K = self.gaussian_kernel(X, X)
        ns, nf = X.shape
        
        P = np.block([[K, -K], [-K, K]])
        P = 0.5 * (P + P.T)
        q = np.hstack([self.epsilon - y, self.epsilon + y])
        A = np.hstack([np.ones(ns), -np.ones(ns)]).reshape(1, -1)
        b = np.array([0.0])
        
        G = np.vstack([np.eye(2 * ns), -np.eye(2 * ns)]).astype(np.float64)
        h = np.hstack([self.C * np.ones(2 * ns), np.zeros(2 * ns)]).astype(np.float64)
        
        P = cvxopt.matrix(P, tc='d')
        q = cvxopt.matrix(q, tc='d')
        A = cvxopt.matrix(A, tc='d')
        b = cvxopt.matrix(b, tc='d')
        G = cvxopt.matrix(G, tc='d')
        h = cvxopt.matrix(h, tc='d')
        
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x']).flatten()
        self.a = alphas[:ns]
        self.a_hat = alphas[ns:]
    
    def get_dual_variables(self):
        return self.a, self.a_hat
    
    def predict(self, X_test):
        K_test = self.gaussian_kernel(X_test, self.X_train)
        y_pred = np.dot(K_test, self.a - self.a_hat)
        support_vectors = np.where((self.a - self.a_hat > 1e-5))[0]
        b = np.mean(self.Y_train[support_vectors] - y_pred[support_vectors]) if len(support_vectors) > 0 else 0
        return y_pred + b

def plot_results(Y_test, Y_pred, moving_avg, t, gamma):
    plt.figure()
    plt.plot(Y_test, label='Actual Closing Price', color='blue')
    plt.plot(Y_pred, label='Predicted Closing Price', color='red', linestyle='dashed')
    plt.plot(moving_avg, label=f'Average Price', color='black', linestyle='dotted')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title(f'SVR Prediction (t={t}, gamma={gamma})')
    plt.legend()
    plt.show()

gamma_values = [1, 0.1, 0.01, 0.001]
for t in [7, 30, 90]:
    for gamma in gamma_values:
        svr_gauss = SlackGaussianSVRDual(C=1.0, epsilon=0.1, gamma=gamma)
        X, Y = create_data(closing_prices, t)
        print(X.shape, Y.shape)
        X_train = X[:int(0.5 * len(X))]
        Y_train = Y[:int(0.5 * len(Y))]
        X_test = X[int(0.5 * len(X)):]
        Y_test = Y[int(0.5 * len(Y)):]
        
        svr_gauss.fit(X_train, Y_train)
        Y_pred = svr_gauss.predict(X_test)
        
        Y_pred = scaler.inverse_transform(Y_pred.reshape(-1, 1)).flatten()
        Y_test = scaler.inverse_transform(Y_test.reshape(-1, 1)).flatten()
        moving_avg = np.convolve(Y_test.flatten(), np.ones(t) / t, mode='valid')
        
        plot_results(Y_test, Y_pred, moving_avg, t, gamma)