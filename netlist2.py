##################################################
######### Netlist for LtSpice simulation #########
##################################################


##################################################
#                   Parameters                                     
#
# 1. N_x = N_y, size of the square lattic 
#
# 2. t0,t,m hopping paramteters, provided 
#       as input by the user (in order)
#
# 3. Input point of the external source
#       propted as 'xys', s=0 for 'a' and 1 for 'b'
# 
# In total we have 6 user input vatiables 
# 
#
# 
#                Netlist Nomenclature 
# 
# 1. loop(lattice cite): 'X #x x #y y s' \
#       (blank space only for visulisation)
#       where #x(y) = x(y) cell index
#       's' = 'a' or 'b' denoted the sublattice index 
# 
# 2. nodes of a lattice cite indexed by j \in 1,2,\dots J
#       are denoted by 'X #x x #y y s j'
# 
##################################################


import numpy as np 
import pandas as pd
import sys , math , os
import PyLTSpice as spice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Cell: # A lattice point == a inductor loop 
    def __init__(self,x,y,s,J):
        self.name = 'x'+str(x)+'x'+str(y)+'y'+str(s)
        self.x = x 
        self.y = y 
        self.s = s 
        self.nodes = np.char.add( np.full(J,self.name) \
                    , np.arange(1,J+1,1).astype('U'))
        tempstr = '' 
        for i in self.nodes:
            tempstr = tempstr + i + ' '
        self.jointnodes = tempstr 
        
class Node: # collection of all lattice points 
    def __init__( self , N , J):
        self.list = np.empty((N,N,2) , dtype = Cell )
        for x in range(0,N):
            for y in range(0,N):
                self.list[x , y,0] = Cell( x +1 , y + 1 , 'a' , J)
                self.list[x , y,1] = Cell( x +1 , y + 1 , 'b' , J)

                

lg = 1e-5 # ground inductance 
l = 1e-5 # inducatance of each capacitor in the loop
c = 1e-7 
N  = int(sys.argv[1]) 
t0 = float(sys.argv[2])    
t  = float(sys.argv[3])
m  = float(sys.argv[4])
omega = ( math.sqrt(((1/lg)+(2/l))/(c*(3*t0/2+3*t/2+m))))
freq = omega / ( 2 * math.pi )
inCoordinate = (int(sys.argv[5]) -1 ,int(sys.argv[6])- 1 ,\
    int(sys.argv[7])) # tuple of the form 'xys'

tempStr = 'Rser=0 Rpar=0 Cpar=0' 

node = Node( N , 4 )
inCell = node.list[inCoordinate]

def printLoop( J , l ): # prints spice subcircuit for the inductor loop
        print('*Sibcircuit Definition')
        print(".subckt loop " + \
            np.array2string(np.arange(1,J+1,1))[1:-1]) 
        for i in range(1 , J): 
            print('l'+str(i),str(i),str(i+1),l)
            print('lg'+str(i),str(i),0,lg)
        print('l'+str(J),str(J),str(1),l)
        print('lg'+str(J),str(J),0,lg)
        print('.ends\n')

def creatLatticeCite(J,node):
    print('* Lattice Cite Definitions')
    for cell in node.list.flatten('C'):
        print( cell.name , cell.jointnodes + 'loop' )


def nodeConnect(cell1 , cell2 ,  J , m , t):
    cap = t*c 
    print('\n*Hopping connection between '\
        +'('+str(cell1.x)+','+str(cell1.y)+') and '\
            +'('+str(cell2.x)+','+str(cell2.y)+')')
    for j in range( 0 , J ):
        print('C'+ cell1.name + cell2.name + str(j) , cell1.nodes[j] \
              , cell2.nodes[(j+m)%J] , str(cap)  )
        
def printConnections( node , N , J , t0 , t , m ):
    for y in range ( 0 , N - 1): 
        for x in range( 0 , N -1 ):
            cella = node.list[ x , y , 0]
            cellb = node.list[ x , y , 1]
            # 1. intracell hopping ~ m 
            nodeConnect( cella , cellb , J , 2, m )
            # 2. intercell in x direction 
            nodeConnect(cella,node.list[x+1,y,1],J,0,t0/2)
            nodeConnect(cellb,node.list[x+1,y,0],J,0,t0/2)
            nodeConnect(cella,node.list[x+1,y,0],J,1,t/2)
            nodeConnect(cellb,node.list[x+1,y,1],J,1,t/2)
            # 3. intercell in the y direction 
            # nodeConnect(cella,node.list[x,y+1,1],J,0,t0/2)
            nodeConnect(cellb,node.list[x,y+1,0],J,0,(t0+t)/2)
    for y in range( 0 , N - 1 ):
        cella = node.list[ N-1 , y , 0]
        cellb = node.list[ N-1 , y , 1]
        nodeConnect( cella , cellb , J , 2, m )
        nodeConnect(cellb,node.list[N-1,y+1,0],J,0,(t0+t)/2)
    for x in range( 0 , N - 1 ):
        cella = node.list[ x , N-1 , 0]
        cellb = node.list[ x , N-1 , 1]  
        nodeConnect( cella , cellb , J , 2, m )
        nodeConnect(cella,node.list[x+1,y,1],J,0,t0/2)
        nodeConnect(cellb,node.list[x+1,y,0],J,0,t0/2)
        nodeConnect(cella,node.list[x+1,y,0],J,1,t/2)
        nodeConnect(cellb,node.list[x+1,y,1],J,1,t/2)  
    nodeConnect( node.list[ N -1 , N-1 , 0] , \
        node.list[ N -1 , N-1 , 1] , J , 2 , m )
    
def boundaryGroundNodes( node , N , J , t0 , t ):
    print("\n\n* Boundary grounding Nodes")
    for x in range(0 , N  ):
        y = 0 
        cell = node.list[ x, y , 0]
        for i in cell.nodes:
            print('c'+str(x)+'x'+str(y)+'y'+i, i ,\
                '0' , str((t+t0)*c/2))
        y = N - 1
        cell = node.list[ x, y , 1]
        for i in cell.nodes:
            print('c'+str(x)+'x'+str(y)+'y'+i, i ,\
                '0' , str((t+t0)*c/2)  )
    for y in range( 0 , N ):
        for x in [ 0 , N-1 ]: 
            for z in [ 0 , 1 ]:
                cell = node.list[ x , y , z ] 
                for i in cell.nodes :
                    print('c'+str(y)+'y'+str(x)+'x'+str(z)+i, \
                        i , '0' , str(t0*c/2) )
                    print('c'+str(y)+'y'+str(x)+'x'+i+str(z), \
                        i , '0' , str(t*c/2) )
    print('\n\n.END')
    
def collectImpedanceData( node , N , freq ,inCoordinate):
    z = np.zeros((N,N))
    if inCoordinate[2] == 0:
        s = 1 
    else:
        s = 0 
    cell2 = node.list[inCoordinate]
    net = spice.SpiceEditor('netlist.net')
    runner = spice.SimRunner()
    for x in range( 0 , N ):
        for y in range( 0 , N ):
            cell1 = node.list[ x ,y , s ]
            trace = ['V('+cell1.nodes[0]+')',\
                'V('+cell2.nodes[0]+')' ]
            net.add_instructions('\n\n*Source and Directive'\
                ,'I1 '+cell1.nodes[0]+' '+cell2.nodes[0]+\
                    ' AC 1 0' ,'.ac lin 1 '+str(freq)+' '\
                        +str(freq))
            raw , log = runner.run_now(net,switches=None, \
                run_filename= 'file.net')
            rawData = spice.RawRead('file.raw' , \
                traces_to_read=trace)
            subdata = rawData.to_dataframe()
            z[x,y] = np.absolute( subdata[trace[0]] - subdata[trace[1]])[0]
            net.reset_netlist()
    return z 

def print_netlist(node , l , J , t0 , t , m ):
    initialStdout = sys.stdout 
    with open( 'netlist.net' , 'w') as f :
        sys.stdout = f 
        printLoop(4,l)
        creatLatticeCite(J,node)
        printConnections( node , N , J , t0 , t , m )
        boundaryGroundNodes( node , N  , 4 , t0 , t )
    sys.stdout = initialStdout

print_netlist( node , l , 4 , t0 , t , m ); 
z = collectImpedanceData( node , N , freq , inCoordinate )
print(z)
plt.imshow( z )
plt.show()
np.save( 'z2', z )