# import numpy as np 
# import pandas as pd 
# import PyLTSpice as spice 
# import matplotlib.pyplot as plt 

# # class Cell: # A lattice point == a inductor loop 
# #     def __init__(self,x,y,s,J):
# #         self.name = 'X'+str(x)+'x'+str(y)+'y'+str(s)
# #         self.x = x 
# #         self.y = y 
# #         self.s = s 
# #         self.nodes = np.char.add( np.full(J,self.name) \
# #                     , np.arange(1,J+1,1).astype('U'))
# #         tempstr = '' 
# #         for i in self.nodes:
# #             tempstr = tempstr + i + ' '
# #         self.jointnodes = tempstr 
        
# # class Node: # collection of all lattice points 
# #     def __init__( self , N , J):
# #         self.list = np.empty((N,N,2) , dtype = Cell )
# #         for x in range(0,N):
# #             for y in range(0,N):
# #                 self.list[x , y,0] = Cell( x +1 , y + 1 , 'a' , J)
# #                 self.list[x , y,1] = Cell( x +1 , y + 1 , 'b' , J)

# # node = Node( 5 , 4 )

# # cell1 = node.list[0,0,0]
# # cell2 = node.list[4,4,1] 

# # f = 147348.84529263328 

# # net = spice.SpiceEditor('netlist.net')
# # net.add_instructions('\n\n*Source and Directive',\
# #     'I1 '+cell1.nodes[0]+' '+cell2.nodes[0]+' SINE(0 1 '+str(f)+')' , \
# #         '.tran '+str(10/f))
# # net.write_netlist('meow.net')
# # runner = spice.SimRunner()
# # raw , log = runner.run_now(net,switches=None, \
# #     run_filename= 'file.net')

# [[ 19.67107343  20.58963108  22.4467715   14.80764614   8.12670564
#    11.1418567   19.39926178  27.2474637   29.31816933  20.95687194
#    10.65210798   8.87293415  13.63851906  12.31527772  33.76667366]
#  [ 24.31733651  30.23221479  21.77752306  16.66777461  20.80504232
#    38.24054389  50.44825106  44.34222814  39.94866255  40.60834866
#    45.48923747  45.35488502  45.96379796  21.95522452  34.3455587 ]
#  [ 44.05238248  29.09005581   5.67646037   1.62565873  13.37400908
#    52.13606172  68.73297196  50.92365685  35.60820593  20.8634801
#    20.26325539  21.78908244  55.14174551  53.03906811   1.2574172 ]
#  [ 45.51261921  55.00214433  58.23712661  69.37838066  56.95903825
#    47.82742496  48.30854018  42.43917254  52.42056591  51.55028342
#    41.54634746  25.36972195  75.93539359   4.82914445  34.12247278]
#  [  9.63269725  12.37603148  57.50258507 114.04230158  84.34533413
#    38.50367212  24.41023652  44.71765378  84.30659312  61.55846735
#    28.53493956  19.28638135  76.81504112  40.89089006   4.49859971]
#  [ 26.01755718  20.57392814  40.25684337  77.64861474  51.52532885
#    27.33695005  27.66988265  23.96886147  36.01691944  23.96713381
#    14.29250073  21.83974755  64.71291888   6.31529661  57.16126167]
#  [ 85.39148753  70.39336208  17.69943341  12.75384654   9.3966757
#    40.67081696  57.06084991  30.65020274  24.9878042   24.09856963
#    69.73297794  79.7058392  112.79664342  71.3695324   25.85758132]
#  [ 42.55731768  19.73599502   4.40655653  16.80114525   3.73907796
#    18.53491098  26.80984021   5.04969566  14.52443669  10.69285349
#    23.21787026  40.94790619  45.09082293  26.15908347  35.60586294]
#  [ 14.67769467  12.74655573  37.67443013  65.56010596  37.0003649
#    23.05033895  29.06788236  56.29336635  96.93796151  56.91739663
#    15.53555905  32.64049403  34.18699936  39.89372436  18.83615015]
#  [ 26.37681627  34.37798422  66.46055069  95.46524973  67.77364443
#    64.42143051  61.62582122  70.87700833 117.19437815  86.64309564
#    44.30657207  34.5585388   50.24146075  10.00039186 107.95566081]
#  [ 34.82664193  30.18113384  33.4268905   42.50759419  47.02015835
#    65.03108801  47.77252202  22.20283077  28.61337941  21.17540197
#    28.12446042  46.6075665   10.30292387  78.96627014   9.39922469]
#  [ 65.3356528   64.43444091  38.95107963  32.49133654  48.88361743
#    88.26361362  72.79309867  21.32228796  14.06123979  26.44406285
#    72.5323697  122.65293154   8.35226359  10.55929837  81.56536027]
#  [ 29.35080724  21.85144151  12.13308854   4.86225746   2.47770422
#    19.56925939  26.90130072  22.04584656  18.07613177   9.96988303
#    15.69323237  28.83193505  12.34075816  50.21062903   3.40905079]
#  [  9.27662819  14.71561567  25.47776163  29.79201706  16.9540721
#     9.20301923  15.66017806  36.3251093   51.94400637  31.4807963
#     6.43913006   7.28280881  23.54852232   6.51245447  62.56434373]
#  [ 30.59609986  22.09961936  21.74354565  27.03582791  32.27555546
#    38.31975273  45.65209533  54.04073084  61.32936672  57.16303247
#    37.02839942  24.63530039  34.14726265  42.36734255  12.97389367]]

