import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations, permutations

class ElementaryCA:
    
    def __init__(self,rule):
        self.rule = rule;
        self.rule_binary = [int(x)for x in f'{rule:#010b}'[-1:1:-1]]
        
        
        
    def evolve(self,width=100,height= 100,init_state='point'):
        self.time_history = np.zeros([height,width])
        
        if init_state == 'point':
            self.time_history[0,width//2]=1;
        if init_state== 'random':
            self.time_history[0,:] = np.round(np.random.rand(1,width))
        if init_state == 'right':
            self.time_history[0,width-1]=1;
        #evolve system over time
        for t in range(1,height):
            for i in range(1,width):
                neighbourhood = self.time_history[t-1,i-1:i+2]
                idx = self.binlist2dec(neighbourhood)
                self.time_history[t,i] = self.rule_binary[idx]
        
    def show(self,ax = None,showTitle=True):
        flag = 0
        if ax is None:
            flag = 1;
            fig = plt.figure(figsize=(8, 6), dpi=150)
            ax = fig.add_subplot(1,1,1)
        ax.imshow(self.time_history,cmap='Greys',interpolation='none')
        ax.axis('off')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.get_xaxis().set_ticks([])
        if showTitle:
            ax.set_title(f"R{self.rule}")
        
        if flag:
            plt.show()
        
                
    def binlist2dec(self,binary,inverted=-1):
        return int(sum([x*2**i for i,x in enumerate(binary[::inverted])]))
                

        
        
def binlist2dec(binary,inverted=-1):
    return sum([x*2**i for i,x in enumerate(binary[::inverted])])

def rulebin(rule):
    return [int(x)for x in f'{rule:#010b}'[-1:1:-1]]

def getPermutations(rule):
    #gets all equivalent permutations of a Rule by permuting inputs
    ruleb = rulebin(rule) #creates binary representation by rule number
    rules ={rule}
    
    for perm in permutations([0,1,2]):
        idx = list(range(8))
        for i in range(8):
            ibin = [int(x) for x in f'{i:#05b}'[2:]] #converts index to binary index
            permed = [ibin[j] for j in perm]
            newibin = binlist2dec(permed)
            idx[i] = newibin

        newruleb = [ruleb[i] for i in idx]
        newrule = binlist2dec(newruleb,inverted=1)
        
        
        rules.add(newrule)
    return rules
def getGEC(rule):
    ruleb = rulebin(rule) #creates binary representation by rule number
    rules ={rule}
    
    for neg in range(8):
        idx = list(range(8)) 
        negb = [int(x) for x in f'{neg:#05b}'[2:]] #converts index to binary index
        for i in range(8):
            ibin = [int(x) for x in f'{i:#05b}'[2:]] #converts index to binary index
            negged = [int(not j) if negb[i] else j for i,j in enumerate(ibin) ] #negate arguments according to neg
            newibin = binlist2dec(negged) #sort ibin by new indices
            idx[i] = newibin
        newruleb = [ruleb[i] for i in idx]
        newrule = binlist2dec(newruleb,inverted=1)
        perms = getPermutations(newrule)
        rules.update(perms)
    return tuple(sorted(list(rules)))

def decimal_to_truth_table(decimal):
    # Convert decimal to binary
    binary = f'{decimal:#010b}'[-1:1:-1]
    # Create a dictionary to store the truth values of each boolean function
    truth_table = {}
    # Iterate through the binary digits
    for i in range(8):
        # Assign the i-th digit to the i-th boolean function
        truth_table[i] = int(binary[i])
    return truth_table

def truth_table_to_dict(truth_table):
    # Create a dictionary to store the function values
    function_dict = {}
    # Iterate through all possible input combinations (i, j, k)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                input_tuple = (i, j, k)
                index = i*4+j*2+k
                function_dict[input_tuple] = truth_table[index]
    return function_dict

def dict_to_lists(function_dict):
    true_inputs = []
    false_inputs = []
    for inputs, output in function_dict.items():
        if output:
            true_inputs.append(np.array(inputs))
        else:
            false_inputs.append(np.array(inputs))
    return true_inputs, false_inputs

