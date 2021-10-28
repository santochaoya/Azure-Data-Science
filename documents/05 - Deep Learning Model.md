# Deep Neural Network Model



## Concepts

* ==**Input Layer**==

  Each neuron represents a feature which the model will process.

* ==**Hidden Layer**==

  

* ==**Output Layer**==

  Each neuron represents an output label.

* ==**Epochs**==

  The iterations of a deep neural network model.



## Train Deep Neural Network Model with PyTorch

### Steps

1. For the first epoch, initialized to choose a weight (**w**) and bias (**b**) randomly. Submit the features for observations which have known labels to the input layer. Generally, the observations are grouped into *batches*.
2. Apply functions to the neuron, if activated, pass the result to the next layer until the output layer produce the predictions.
3. Calculate the variance(**Loss**) between actual and predicted label.
4. Revised the weight and bias to find the lower **Loss**. There adjustments are *backpropagated* to the neurons in the network layers.
5. Next epoch with revised weight and bias values, repeats the batch.



> ### Preparation of dataset
>
> This document will get dataset from the Palmer Islands penguins dataset, which contains observations of three different species of penguins:  
>
> * 0 : *Adelie* 阿德利企鹅
>
>   ![Adelie](C:\Users\Xiao_Meng\OneDrive - EPAM\Dodumentations\Images\Adelie.jpg)
>
> * 1 : *Gentoo* 白眉企鹅
>
>   ![Gentoo](C:\Users\Xiao_Meng\OneDrive - EPAM\Dodumentations\Images\Gentoo.jpg)
>
> * 2 : *Chinstrap* 帽带企鹅
>
>   ![Chinstrap](C:\Users\Xiao_Meng\OneDrive - EPAM\Dodumentations\Images\Chinstrap.jpg)



### Training

dataset read as variable : `penguins`



#### Explore the Dataset

Deep learning models work best when the features are on similar scales.



#### Split the training and validations datasets

```python
from sklearn.model_selection import train_test_split

features = ['CulmenLength','CulmenDepth','FlipperLength','BodyMass']
label = 'Species'

# Split the dataset into training and test dataset from 73% - 30%
x_train, x_test, y_train, y_test = train_test_split(penguins[features].values,
													penguins[label].values,
													test_size=0.30,
													random_state=0)
						
print('Training Set: %d, Test Set: %d \n' % (len(x_train), len(x_test)))
```



#### Install and import the PyTorch 

```powershell
$pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

```python
import torch
import torch.nn as nn
import torch.utils.data as td

# Set random seed for reproducability
torch_manual_seed(0)

print('Library imported - ready to use PyTorch', torch.__version__)
```



#### Prepare the data for PyTorch

PyTorch makes use of data loaders to load the training and validation data in batches.

* Wrap numpy arrays  in PyTorch datasets

  ```python
  torch.Tensor(data)
  ```

  

* Create loaders to read batches from PyTorch datasets

  ```python
  td.DataLoader(data, batch_size)
  ```

  which load data from parameter ```data``.



```python
# Create a dataset and loader for the training data and labels
train_x = torch.Tensor(x_train).float()
train_y = torch.Tensor(y_train).long()
train_ds = td.TensorDataset(train_x, train_y)
train_loader = td.DataLoader(train_ds, batch_size=20,
	shuffle=False, num_workers=1)

# Create a dataset and loader for the test
test_x = torch.Tensor(x_test).float()
test_y = torch.tensor(y_test).long()
test_ds = td.TensorDataset(test_x, te_y)
test_loader = td.DataLoader(test_ds, batch_size=20,
	shuffle=False, num_workers=1)
```

> **TODO**: DataLoader()

#### Define a neural network

We will define a neural network model with 3 fully-connected layers with activation function ReLU.

* **Input layer** : receive input value for each feature with activation function ReLU
* **Hidden Layer**: apply ReLU on each neuron
* **Output Layer**: generate non-negative numeric output for each classification.

Activation Function:

* **ReLU**: 

$$
f(x) = max(0, x)
$$

![DNN1](C:\Users\Xiao_Meng\OneDrive - EPAM\Dodumentations\Images\DNN1.PNG)

```python
# Number of nodes in hidden layer
hl = 10

# Define the Neural Network
class PenguinNet(nn.Model):
	def __init__(self):
		super(PenguinNet, self).__init__()
		self.fc1 = nn.Linear(len(features), hl)
		self.fc2 = nn.Linear(hl, hl)
		self.fc3 = nn.Linear(hl, len(penguin))
		
    def forward(self, x):
    	x = torch.relu(self.fc1(x))
    	x = torch.relu(self.fc2(x))
    	x = torch.relu(self.fc3(x))
    	
# Create a model instance from the network
model = PenguinNet()
```



#### Train the model

Input ```data_loader``` with form as: training data and labels with all batches.

```python
def train(model, data_loader, optimizer):
	# Set the model to training mode
	model.train()
	train_loss = 0
	
	for batch, tensor in enumerate(data_loader):
		data, target = tensor
		
		# feed forward
		optimizer.zero_grad()
		out = model(data)
		loss = loss_criteria(out, target)
		train_loss += loss.item()
		
		# backpropagrate
		loss.backward()
		optimizer.step()
		
    # Return average loss
    avg_loss = train_loss / (batch + 1)
    print('Training set: Average loss: {:.6f}'.format(avg_loss))
    
    return avg_loss
			
```

```python
def test(model, data_loader, optimizer):
	# Set the model to e mode(Not backpropagate)
	model.eval()
	test_loss = 0
	correct = 0
	
	with torch.no_grad():
		batch_count = 0
		for batch, tensor in enumerate(data_loader):
			batch_count += 1
			data, target = tensor
			
			# Get the predictions
			out = model(data)
			
			# Calculate teh loss
			test_loss += loss_criteria(out, target).item()
			
			# Calculate the accuracy
			_, predicted = torch.max(out.data, 1)
			correct += torch.sum(target==predicted).item()
			
	# Calculate the average loss and total accuracy for this epoch
	avg_loss = test_loss / batch_count
    print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    avg_loss, correct, len(data_loader.dataset),
    100. * correct / len(data_loader.dataset)))
    
    # return average loss for the epoch
    return avg_loss
	
```

```

```

