import torch
from utils import plotsAnalysis
if __name__ == '__main__':
    # pathnamelist = ['/home/sr365/MM_Bench/MDN/models/Peurifoy/g5',
    #                 '/home/sr365/MM_Bench/MDN/models/Peurifoy/g10',
    #                 '/home/sr365/MM_Bench/MDN/models/Peurifoy/g20',
    #                ]
    pathnamelist = ['/home/sr365/MM_Bench/NA/models/Peurifoy/conv/']
    for pathname in pathnamelist:
        
        # Forward: Convolutional swipe
        #plotsAnalysis.HeatMapBVL('kernel_first','kernel_second','layer vs unit Heat Map',save_name=pathname + '_heatmap.png',
        #                        HeatMap_dir='models/'+pathname,feature_1_name='kernel_first',feature_2_name='kernel_second')
        
        # General: Complexity swipe
        #plotsAnalysis.HeatMapBVL('num_layers','num_unit','layer vs unit Heat Map',save_name=pathname + '_heatmap.png',
        #                        HeatMap_dir=pathname,feature_1_name='linear',feature_2_name='linear_unit')
        
        # General: Reg scale and num_layers
        plotsAnalysis.HeatMapBVL('num_layers','reg_scale','layer vs unit Heat Map',save_name=pathname + '_heatmap.png',
                                HeatMap_dir=pathname,feature_1_name='linear',feature_2_name='reg_scale')
        
        # VAE: kl_coeff and num_layers
        #plotsAnalysis.HeatMapBVL('num_layers','kl_coeff','layer vs unit Heat Map',save_name=pathname + '_heatmap.png',
        #                        HeatMap_dir=pathname,feature_1_name='linear_d',feature_2_name='kl_coeff')

        # VAE: kl_coeff and dim_z
        #plotsAnalysis.HeatMapBVL('dim_z','kl_coeff','layer vs unit Heat Map',save_name=pathname + '_heatmap.png',
        #                       HeatMap_dir=pathname,feature_1_name='dim_z',feature_2_name='kl_coeff')

        # VAE: dim_z and num_layers
        #plotsAnalysis.HeatMapBVL('dim_z','num_layers','layer vs unit Heat Map',save_name=pathname + '_heatmap.png',
        #                       HeatMap_dir=pathname,feature_1_name='dim_z',feature_2_name='linear_d')

        # General: Reg scale and num_unit (in linear layer)
        #plotsAnalysis.HeatMapBVL('reg_scale','num_unit','layer vs unit Heat Map',save_name=pathname + '_heatmap.png',
        #                        HeatMap_dir=pathname,feature_1_name='reg_scale',feature_2_name='linear_unit')
        
        # cINN or INN: Couple layer num and lambda mse
        #plotsAnalysis.HeatMapBVL('couple_layer_num','lambda_mse','layer vs unit Heat Map',save_name=pathname + '_heatmap.png',
        #                        HeatMap_dir=pathname,feature_1_name='couple_layer_num',feature_2_name='lambda_mse')
        
        # INN: Couple layer num and dim_pad
        #plotsAnalysis.HeatMapBVL('couple_layer_num','dim_tot','layer vs unit Heat Map',save_name=pathname + '_heatmap.png',
        #                        HeatMap_dir=pathname,feature_1_name='couple_layer_num',feature_2_name='dim_tot')

        # INN: Lambda_mse num and dim_pad
        #plotsAnalysis.HeatMapBVL('couple_layelambda_mser_num','dim_tot','lambda_mse vs dim_tot Heat Map',save_name=pathname + '_heatmap.png',
        #                        HeatMap_dir=pathname,feature_1_name='lambda_mse',feature_2_name='dim_tot')
        
        # INN: Couple layer num and dim_z
        #plotsAnalysis.HeatMapBVL('couple_layer_num','dim_z','layer vs unit Heat Map',save_name=pathname + '_heatmap.png',
        #                        HeatMap_dir=pathname,feature_1_name='couple_layer_num',feature_2_name='dim_z')
