import torch
from utils import plotsAnalysis
if __name__ == '__main__':
    pathnamelist = ['/home/sr365/MM_Bench/Tandem/models/Peurifoy/']
    #pathnamelist = ['/home/sr365/MM_Bench/cINN/models/Chen/Chen_reg1e-3',
    #                '/home/sr365/MM_Bench/cINN/models/Chen/Chen_reg5e-3',
    #                '/home/sr365/MM_Bench/cINN/models/Chen/Chen_reg1e-4',
    #                '/home/sr365/MM_Bench/cINN/models/Chen/Chen_reg5e-4']
    for pathname in pathnamelist:
        
        # Forward: Convolutional swipe
        #plotsAnalysis.HeatMapBVL('kernel_first','kernel_second','layer vs unit Heat Map',save_name=pathname + '_heatmap.png',
        #                        HeatMap_dir='models/'+pathname,feature_1_name='kernel_first',feature_2_name='kernel_second')
        
        # General: Complexity swipe
        plotsAnalysis.HeatMapBVL('num_layers','num_unit','layer vs unit Heat Map',save_name=pathname + '_heatmap.png',
                                HeatMap_dir=pathname,feature_1_name='linear_b',feature_2_name='linear_unit')
        
        # General: Reg scale and num_layers
        #plotsAnalysis.HeatMapBVL('num_layers','reg_scale','layer vs unit Heat Map',save_name=pathname + '_heatmap.png',
        #                        HeatMap_dir=pathname,feature_1_name='linear',feature_2_name='reg_scale')
        
        # General: Reg scale and num_unit (in linear layer)
        #plotsAnalysis.HeatMapBVL('reg_scale','num_unit','layer vs unit Heat Map',save_name=pathname + '_heatmap.png',
        #                        HeatMap_dir=pathname,feature_1_name='reg_scale',feature_2_name='linear_unit')
        
        # cINN or INN: Couple layer num and lambda mse
        #plotsAnalysis.HeatMapBVL('couple_layer_num','lambda_mse','layer vs unit Heat Map',save_name=pathname + '_heatmap.png',
        #                        HeatMap_dir=pathname,feature_1_name='couple_layer_num',feature_2_name='lambda_mse')
        
        # INN: Couple layer num and dim_pad
        #plotsAnalysis.HeatMapBVL('couple_layer_num','dim_tot','layer vs unit Heat Map',save_name=pathname + '_heatmap.png',
        #                        HeatMap_dir=pathname,feature_1_name='couple_layer_num',feature_2_name='dim_tot')
        
        # INN: Couple layer num and dim_z
        #plotsAnalysis.HeatMapBVL('couple_layer_num','dim_z','layer vs unit Heat Map',save_name=pathname + '_heatmap.png',
        #                        HeatMap_dir=pathname,feature_1_name='couple_layer_num',feature_2_name='dim_z')
