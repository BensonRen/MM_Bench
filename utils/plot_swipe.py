import torch
from utils import plotsAnalysis
if __name__ == '__main__':
    #pathnamelist = ['/home/sr365/MM_Bench/Tandem/models/Yang/sweep3/noconv/',
    #               '/home/sr365/MM_Bench/Tandem/models/Yang/sweep3/conv_444_334_112/',
    #                ]
    pathnamelist = ['/home/sr365/MM_Bench/NA/models/Chen/sweep5/lr0.001/reg0/',
                    '/home/sr365/MM_Bench/NA/models/Chen/sweep5/lr0.0001/reg0/']
    #pathnamelist= ['/home/sr365/MM_Bench/NA/models/Yang_sim/conv_444_435_211/']
    for pathname in pathnamelist:
        
        # Forward: Convolutional swipe
        #plotsAnalysis.HeatMapBVL('kernel_first','kernel_second','layer vs unit Heat Map',save_name=pathname + '_heatmap.png',
        #                        HeatMap_dir='models/'+pathname,feature_1_name='kernel_first',feature_2_name='kernel_second')
        
        # General: Complexity swipe
        plotsAnalysis.HeatMapBVL('num_layers','num_unit','layer vs unit Heat Map',save_name=pathname + 'layer vs unit_heatmap.png',
                                 HeatMap_dir=pathname,feature_1_name='linear',feature_2_name='linear_unit')
        
        # General: lr vs layernum
        #plotsAnalysis.HeatMapBVL('num_layers','lr','layer vs unit Heat Map',save_name=pathname + 'layer vs lr_heatmap.png',
        #                         HeatMap_dir=pathname,feature_1_name='linear',feature_2_name='lr')

        # MDN: num layer and num_gaussian
        #plotsAnalysis.HeatMapBVL('num_layers','num_gaussian','layer vs num_gaussian Heat Map',save_name=pathname + 'layer vs num_gaussian heatmap.png',
        #                         HeatMap_dir=pathname,feature_1_name='linear',feature_2_name='num_gaussian')
        
        # General: Reg scale and num_layers
        #plotsAnalysis.HeatMapBVL('num_layers','reg_scale','layer vs reg Heat Map',save_name=pathname + 'layer vs reg_heatmap.png',
        #                       HeatMap_dir=pathname,feature_1_name='linear',feature_2_name='reg_scale')
        
        # VAE: kl_coeff and num_layers
        #plotsAnalysis.HeatMapBVL('num_layers','kl_coeff','layer vs kl_coeff Heat Map',save_name=pathname + 'layer vs kl_coeff_heatmap.png',
        #                        HeatMap_dir=pathname,feature_1_name='linear_d',feature_2_name='kl_coeff')

        # VAE: kl_coeff and dim_z
        #plotsAnalysis.HeatMapBVL('dim_z','kl_coeff','kl_coeff vs dim_z Heat Map',save_name=pathname + 'kl_coeff vs dim_z Heat Map heatmap.png',
         #                      HeatMap_dir=pathname,feature_1_name='dim_z',feature_2_name='kl_coeff')

        # VAE: dim_z and num_layers
        #plotsAnalysis.HeatMapBVL('dim_z','num_layers','layer vs unit Heat Map',save_name=pathname + 'layer vs dim_z Heat Map heatmap.png',
         #                     HeatMap_dir=pathname,feature_1_name='dim_z',feature_2_name='linear_d')
        
        # VAE: dim_z and num_unit
        #plotsAnalysis.HeatMapBVL('dim_z','num_unit','dim_z vs unit Heat Map',save_name=pathname + 'dim_z vs unit Heat Map heatmap.png',
         #                      HeatMap_dir=pathname,feature_1_name='dim_z',feature_2_name='linear_unit')

        # General: Reg scale and num_unit (in linear layer)
        #plotsAnalysis.HeatMapBVL('reg_scale','num_unit','reg_scale vs unit Heat Map',save_name=pathname + 'reg_scale vs unit_heatmap.png',
        #                        HeatMap_dir=pathname,feature_1_name='reg_scale',feature_2_name='linear_unit')
        
        # cINN or INN: Couple layer num and lambda mse
        #plotsAnalysis.HeatMapBVL('couple_layer_num','lambda_mse','couple_num vs lambda mse Heat Map',save_name=pathname + 'couple_num vs lambda mse_heatmap.png',
        #                       HeatMap_dir=pathname,feature_1_name='couple_layer_num',feature_2_name='lambda_mse')
        
        # # cINN or INN: Couple layer num and reg scale
        # plotsAnalysis.HeatMapBVL('couple_layer_num','reg_scale','layer vs unit Heat Map',save_name=pathname + '_heatmap.png',
        #                         HeatMap_dir=pathname,feature_1_name='couple_layer_num',feature_2_name='reg_scale')
        
        # # INN: Couple layer num and dim_pad
        # plotsAnalysis.HeatMapBVL('couple_layer_num','dim_tot','couple_layer_num vs dim pad Heat Map',save_name=pathname + 'couple_layer_num vs dim pad _heatmap.png',
        #                         HeatMap_dir=pathname,feature_1_name='couple_layer_num',feature_2_name='dim_tot')

        # # INN: Lambda_mse num and dim_pad
        # plotsAnalysis.HeatMapBVL('lambda_mse','dim_tot','lambda_mse vs dim_tot Heat Map',save_name=pathname + 'lambda_mse vs dim_tot_heatmap.png',
        #                         HeatMap_dir=pathname,feature_1_name='lambda_mse',feature_2_name='dim_tot')
        
        # # INN: Couple layer num and dim_z
        # plotsAnalysis.HeatMapBVL('couple_layer_num','dim_z','couple_layer_num vs dim_z Heat Map',save_name=pathname + 'couple_layer_num vs dim_z_heatmap.png',
        #                         HeatMap_dir=pathname,feature_1_name='couple_layer_num',feature_2_name='dim_z')
