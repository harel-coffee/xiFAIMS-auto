#normal reg
python ../xirt_analysis.py ../data/4PM_DSS_LS_nonunique1pCSM.csv ../parameters/xirt_faims_reg_aux1_w10.yaml ../parameters/xirt_learning.yaml 128 50
python ../xirt_analysis.py ../data/4PM_DSS_LS_nonunique1pCSM.csv ../parameters/xirt_faims_reg_aux1_w100.yaml ../parameters/xirt_learning.yaml 128 50
												
#normal ordinal                                             
python ../xirt_analysis.py ../data/4PM_DSS_LS_nonunique1pCSM.csv ../parameters/xirt_faims_ordinal_aux1.yaml ../parameters/xirt_learning.yaml 128 50
python ../xirt_analysis.py ../data/4PM_DSS_LS_nonunique1pCSM.csv ../parameters/xirt_faims_ordinal_aux1_w10.yaml ../parameters/xirt_learning.yaml 128 50
python ../xirt_analysis.py ../data/4PM_DSS_LS_nonunique1pCSM.csv ../parameters/xirt_faims_ordinal_aux1_w100.yaml ../parameters/xirt_learning.yaml 128 50
