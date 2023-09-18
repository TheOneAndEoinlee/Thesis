import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations, permutations
from matplotlib.animation import FuncAnimation
from numpy.ma import masked_array



class ElementaryCA:
    
    def __init__(self,rule):
        self.rule = rule
        self.rule_binary = [int(x)for x in f'{rule:#010b}'[-1:1:-1]]
        
        
        
    def evolve(self,width=100,height= 100,init_state='point'):
        self.time_history = np.zeros([height,width])
        
        if init_state == 'point':
            self.time_history[0,width//2]=1
        if init_state== 'random':
            self.time_history[0,:] = np.round(np.random.rand(1,width))
        if init_state == 'right':
            self.time_history[0,width-1]=1
        #evolve system over time
        for t in range(1,height):
            for i in range(1,width):
                neighbourhood = self.time_history[t-1,i-1:i+2]
                idx = self.binlist2dec(neighbourhood)
                self.time_history[t,i] = self.rule_binary[idx]
        
    def show(self,ax = None,showTitle=True):
        flag = 0
        if ax is None:
            flag = 1
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
    def animate(self, interval=50):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        ax.axis('off')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.get_xaxis().set_ticks([])
        # This will display the first row to start with
        # Create a masked array where all elements are initially masked
        # Create a masked version of time history
        masked_data = np.ma.masked_array(self.time_history, mask=True)
        masked_data.mask[0, :] = False
        
        im = ax.imshow(masked_data, cmap='Greys', interpolation='none')
        
        def update(frame):
            # Unmask one additional row
            masked_data.mask[frame, :] = False
            im.set_array(masked_data)
            return [im]

        ani = FuncAnimation(fig, update, frames=self.time_history.shape[0], 
                            blit=True, interval=interval, repeat=False)

        return ani
    
    def animatepcm(self, interval=50):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        ax.axis('off')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.get_xaxis().set_ticks([])

        # Create a masked array where all elements are initially masked
        masked_data = np.ma.masked_array(self.time_history, mask=True)
        masked_data.mask[0, :] = False

        # Define the grid for pcolormesh
        y, x = np.mgrid[slice(0, masked_data.shape[0] + 1),
                        slice(0, masked_data.shape[1] + 1)]
        
        im = ax.pcolormesh(x, y, masked_data, shading='auto', cmap='Greys')
        
        def update(frame):
            # Unmask one additional row
            masked_data.mask[frame, :] = False
            im.set_array(masked_data[:-1, :-1].ravel())
            return [im]

        ani = FuncAnimation(fig, update, frames=self.time_history.shape[0], 
                            blit=True, interval=interval, repeat=False)

        return ani








        
        
def binlist2dec(binary,inverted=-1):
    return sum([x*2**i for i,x in enumerate(binary[::inverted])])

def rulebin(rule):
    return [int(x)for x in f'{rule:#010b}'[-1:1:-1]]

def permuteRule(rule,perm):
    ruleb = rulebin(rule) #creates binary representation by rule number e.g. 110 -> [0,1,1,0,1,1,1,0]
    idx = list(range(8)) #creates index list [0,1,2,3,4,5,6,7]
    for i in range(8):
        ibin = [int(x) for x in f'{i:#05b}'[2:]] #converts index to binary index e.g. 5 -> [1,0,1]
        permed = [ibin[j] for j in perm] #permutes binary index e.g. [1,0,1] -> [0,1,1]
        newibin = binlist2dec(permed) #sort ibin by new indices e.g. [0,1,1] -> 3
        idx[i] = newibin #replace index with new index e.g. 5 -> 3
    newruleb = [ruleb[i] for i in idx] #sort ruleb by new indices
    newrule = binlist2dec(newruleb,inverted=1) #convert binary representation to rule number

    return newrule

def getPermutations(rule):
    #gets all equivalent permutations of a Rule by permuting inputs
    rules ={rule}
    
    for perm in permutations([0,1,2]):
        newrule = permuteRule(rule,perm)
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

def separable_rules():
    ec =   [254, 252, 72, 248, 240,
            22, 232, 126, 30, 6,
            106, 192, 60, 124, 224, 
            104, 62, 183, 129, 128]

    pls=   [[1,1,1],[1,1,0],[2,1,2],[2,1,1],[1,0,0],
            [1,1,1],[1,1,1],[1,1,1],[2,1,1],[2,1,1],
            [1,1,2],[1,1,0],[1,1,0],[2,2,1],[2,1,1],
            [1,1,1],[2,2,1],[2,1,2],[1,1,1],[1,1,1]]

    Ts = [(0.9,3.2),(0.5,2.5),(2.5,3.5),(1.5,4.5),(0.5,1.5),
        (0.5,1.5),(1.5,3.5),(0.8,2.2),(0.75,2.4),(0.5,1.5),
        (1.5,3.5),(1.5,2.5),(0.5,1.5),(1.5,4.5),(2.5,4.5),
        (1.5,2.5),(0.8,3.2),(2.5,3.5),(0.5,2.5),(2.5,3.5)]


    rule_subset= [rule for rule in ec if rule not in [183,129]]
    rules = {rule: (pl,T) for rule,pl,T in zip(ec,pls,Ts) if rule in rule_subset}
    newRules = {}
    ruleSets = []

    shift_left = lambda lst, spaces: lst[spaces % len(lst):] + lst[:spaces % len(lst)]
    shift_right = lambda lst, spaces: lst[-spaces % len(lst):] + lst[:-spaces % len(lst)]

    for i,rule in enumerate(rules.keys()):
        ruleSets.append(set())  # change from dictionary to set
        for j  in range(3):
            perml = shift_left([0,1,2],j)
            permr = shift_right([0,1,2],j)
            newpl = [rules[rule][0][i] for i in permr]
            
            newRule = permuteRule(rule,perml)
            newRules[newRule] = (newpl,rules[rule][1])
            ruleSets[i].add(newRule)  # update to use add method on set
    return newRules




