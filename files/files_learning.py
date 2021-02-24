import learning as learn
import numpy as np

from files import penny_files as p
from files import carmen_files as c
from files import menta_files as m 

#############  penny ########################

# ------------ base features -------------

# x_test1 = learn.get_features(p.sig_test_penny1)
# np.save("test\\penny\\features\\" + 'area_' + str(1) + '_' + "penny" + '_features_extra_test_' + str(len(x_test1)) + '_signals.npy', np.array(x_test1))
# x_test2 = learn.get_features(p.sig_test_penny2)
# np.save("test\\penny\\features\\" + 'area_' + str(2) + '_' + "penny" + '_features_extra_test_' + str(len(x_test2)) + '_signals.npy', np.array(x_test2))
# x_test3 = learn.get_features(p.sig_test_penny3)
# np.save("test\\penny\\features\\" + 'area_' + str(3) + '_' + "penny" + '_features_extra_test_' + str(len(x_test3)) + '_signals.npy', np.array(x_test3))
# x_test4 = learn.get_features(p.sig_test_penny4)
# np.save("test\\penny\\features\\" + 'area_' + str(4) + '_' + "penny" + '_features_extra_test_' + str(len(x_test4)) + '_signals.npy', np.array(x_test4))

# x_train1 = learn.get_features(p.sig_train_penny1)
# np.save("train\\penny\\features\\" + 'area_' + str(1) + '_' + "penny"+ '_features_extra_train' + str(len(x_train1)) + '_signals.npy', np.array(x_train1))
# x_train2 = learn.get_features(p.sig_train_penny2)
# np.save("train\\penny\\features\\" + 'area_' + str(2) + '_' + "penny"+ '_features_extra_train' + str(len(x_train2)) + '_signals.npy', np.array(x_train2))
# x_train3 = learn.get_features(p.sig_train_penny3)
# np.save("train\\penny\\features\\" + 'area_' + str(3) + '_' + "penny"+ '_features_extra_train' + str(len(x_train3)) + '_signals.npy', np.array(x_train3))
# x_train4 = learn.get_features(p.sig_train_penny4)
# np.save("train\\penny\\features\\" + 'area_' + str(4) + '_' + "penny"+ '_features_extra_train' + str(len(x_train4)) + '_signals.npy', np.array(x_train4))



# -------------------- psd ------------------------------
 #------------------welch ----------------------
# x_test1 = learn.get_specific_additional_features(p.sig_test_penny1, "welch")
# np.save("test\\penny\\features\\psd_welch\\" + 'area_' + str(1) + '_' + "penny" + '_features_extra_test_' + str(len(x_test1)) + '_signals.npy', np.array(x_test1))
# x_test2 = learn.get_specific_additional_features(p.sig_test_penny2, "welch")
# np.save("test\\penny\\features\\psd_welch\\" + 'area_' + str(2) + '_' + "penny" + '_features_extra_test_' + str(len(x_test2)) + '_signals.npy', np.array(x_test2))
# x_test3 = learn.get_specific_additional_features(p.sig_test_penny3, "welch")
# np.save("test\\penny\\features\\psd_welch\\" + 'area_' + str(3) + '_' + "penny" + '_features_extra_test_' + str(len(x_test3)) + '_signals.npy', np.array(x_test3))
# x_test4 = learn.get_specific_additional_features(p.sig_test_penny4, "welch")
# np.save("test\\penny\\features\\psd_welch\\" + 'area_' + str(4) + '_' + "penny" + '_features_extra_test_' + str(len(x_test4)) + '_signals.npy', np.array(x_test4))
#
# x_train1 = learn.get_specific_additional_features(p.sig_train_penny1, "welch")
# np.save("train\\penny\\features\\psd_welch\\" + 'area_' + str(1) + '_' + "penny"+ '_features_extra_train' + str(len(x_train1)) + '_signals.npy', np.array(x_train1))
# x_train2 = learn.get_specific_additional_features(p.sig_train_penny2, "welch")
# np.save("train\\penny\\features\\psd_welch\\" + 'area_' + str(2) + '_' + "penny"+ '_features_extra_train' + str(len(x_train2)) + '_signals.npy', np.array(x_train2))
# x_train3 = learn.get_specific_additional_features(p.sig_train_penny3, "welch")
# np.save("train\\penny\\features\\psd_welch\\" + 'area_' + str(3) + '_' + "penny"+ '_features_extra_train' + str(len(x_train3)) + '_signals.npy', np.array(x_train3))
# x_train4 = learn.get_specific_additional_features(p.sig_train_penny4, "welch")
# np.save("train\\penny\\features\\psd_welch\\" + 'area_' + str(4) + '_' + "penny"+ '_features_extra_train' + str(len(x_train4)) + '_signals.npy', np.array(x_train4))

#----------multitaper-------------------
# val = "multitapers"
# x_test1 = learn.get_specific_additional_features(p.sig_test_penny1, val)
# np.save("test\\penny\\features\\psd_multitaper\\" + 'area_' + str(1) + '_' + "penny" + '_features_extra_test_' + str(len(x_test1)) + '_signals.npy', np.array(x_test1))
# x_test2 = learn.get_specific_additional_features(p.sig_test_penny2, val)
# np.save("test\\penny\\features\\psd_multitaper\\" + 'area_' + str(2) + '_' + "penny" + '_features_extra_test_' + str(len(x_test2)) + '_signals.npy', np.array(x_test2))
# x_test3 = learn.get_specific_additional_features(p.sig_test_penny3, val)
# np.save("test\\penny\\features\\psd_multitaper\\" + 'area_' + str(3) + '_' + "penny" + '_features_extra_test_' + str(len(x_test3)) + '_signals.npy', np.array(x_test3))
# x_test4 = learn.get_specific_additional_features(p.sig_test_penny4, val)
# np.save("test\\penny\\features\\psd_multitaper\\" + 'area_' + str(4) + '_' + "penny" + '_features_extra_test_' + str(len(x_test4)) + '_signals.npy', np.array(x_test4))
#
# x_train1 = learn.get_specific_additional_features(p.sig_train_penny1,val)
# np.save("train\\penny\\features\\psd_multitaper\\" + 'area_' + str(1) + '_' + "penny"+ '_features_extra_train' + str(len(x_train1)) + '_signals.npy', np.array(x_train1))
# x_train2 = learn.get_specific_additional_features(p.sig_train_penny2, val)
# np.save("train\\penny\\features\\psd_multitaper\\" + 'area_' + str(2) + '_' + "penny"+ '_features_extra_train' + str(len(x_train2)) + '_signals.npy', np.array(x_train2))
# x_train3 = learn.get_specific_additional_features(p.sig_train_penny3, val)
# np.save("train\\penny\\features\\psd_multitaper\\" + 'area_' + str(3) + '_' + "penny"+ '_features_extra_train' + str(len(x_train3)) + '_signals.npy', np.array(x_train3))
# x_train4 = learn.get_specific_additional_features(p.sig_train_penny4, val)
# np.save("train\\penny\\features\\psd_multitaper\\" + 'area_' + str(4) + '_' + "penny"+ '_features_extra_train' + str(len(x_train4)) + '_signals.npy', np.array(x_train4))

#
# x_test1 = learn.get_additional_features(p.sig_test_penny1)
# np.save("test\\penny\\features\\hilbert\\" + 'area_' + str(1) + '_' + "penny" + '_features_extra_test_' + str(len(x_test1)) + '_signals.npy', np.array(x_test1))
# x_test2 = learn.get_additional_features(p.sig_test_penny2)
# np.save("test\\penny\\features\\hilbert\\" + 'area_' + str(2) + '_' + "penny" + '_features_extra_test_' + str(len(x_test2)) + '_signals.npy', np.array(x_test2))
# x_test3 = learn.get_additional_features(p.sig_test_penny3)
# np.save("test\\penny\\features\\hilbert\\" + 'area_' + str(3) + '_' + "penny" + '_features_extra_test_' + str(len(x_test3)) + '_signals.npy', np.array(x_test3))
# x_test4 = learn.get_additional_features(p.sig_test_penny4)
# np.save("test\\penny\\features\\hilbert\\" + 'area_' + str(4) + '_' + "penny" + '_features_extra_test_' + str(len(x_test4)) + '_signals.npy', np.array(x_test4))
#
# x_train1 = learn.get_additional_features(p.sig_train_penny1)
# np.save("train\\penny\\features\\hilbert\\" + 'area_' + str(1) + '_' + "penny"+ '_features_extra_train' + str(len(x_train1)) + '_signals.npy', np.array(x_train1))
# x_train2 = learn.get_additional_features(p.sig_train_penny2)
# np.save("train\\penny\\features\\hilbert\\" + 'area_' + str(2) + '_' + "penny"+ '_features_extra_train' + str(len(x_train2)) + '_signals.npy', np.array(x_train2))
# x_train3 = learn.get_additional_features(p.sig_train_penny3)
# np.save("train\\penny\\features\\hilbert\\" + 'area_' + str(3) + '_' + "penny"+ '_features_extra_train' + str(len(x_train3)) + '_signals.npy', np.array(x_train3))
# x_train4 = learn.get_additional_features(p.sig_train_penny4)
# np.save("train\\penny\\features\\hilbert\\" + 'area_' + str(4) + '_' + "penny"+ '_features_extra_train' + str(len(x_train4)) + '_signals.npy', np.array(x_train4))








# --------------additional features -----------------------

# x_test1 = learn.get_additional_features(p.sig_test_penny1)
# np.save("test\\penny\\features\\additional_featues\\" + 'area_' + str(1) + '_' + "penny" + '_features_extra_test_' + str(len(x_test1)) + '_signals.npy', np.array(x_test1))
# x_test2 = learn.get_additional_features(p.sig_test_penny2)
# np.save("test\\penny\\features\\additional_featues\\" + 'area_' + str(2) + '_' + "penny" + '_features_extra_test_' + str(len(x_test2)) + '_signals.npy', np.array(x_test2))
# x_test3 = learn.get_additional_features(p.sig_test_penny3)
# np.save("test\\penny\\features\\additional_featues\\" + 'area_' + str(3) + '_' + "penny" + '_features_extra_test_' + str(len(x_test3)) + '_signals.npy', np.array(x_test3))
# x_test4 = learn.get_additional_features(p.sig_test_penny4)
# np.save("test\\penny\\features\\additional_featues\\" + 'area_' + str(4) + '_' + "penny" + '_features_extra_test_' + str(len(x_test4)) + '_signals.npy', np.array(x_test4))

# x_train1 = learn.get_additional_features(p.sig_train_penny1)
# np.save("train\\penny\\features\\additional_featues\\" + 'area_' + str(1) + '_' + "penny"+ '_features_extra_train' + str(len(x_train1)) + '_signals.npy', np.array(x_train1))
# x_train2 = learn.get_additional_features(p.sig_train_penny2)
# np.save("train\\penny\\features\\additional_featues\\" + 'area_' + str(2) + '_' + "penny"+ '_features_extra_train' + str(len(x_train2)) + '_signals.npy', np.array(x_train2))
# x_train3 = learn.get_additional_features(p.sig_train_penny3)
# np.save("train\\penny\\features\\additional_featues\\" + 'area_' + str(3) + '_' + "penny"+ '_features_extra_train' + str(len(x_train3)) + '_signals.npy', np.array(x_train3))
# x_train4 = learn.get_additional_features(p.sig_train_penny4)
# np.save("train\\penny\\features\\additional_featues\\" + 'area_' + str(4) + '_' + "penny"+ '_features_extra_train' + str(len(x_train4)) + '_signals.npy', np.array(x_train4))


# -------------------- psd ------------------------------

# x_test1 = learn.get_specific_additional_features(p.sig_test_penny1)
# np.save("test\\penny\\features\\psd_av\\" + 'area_' + str(1) + '_' + "penny" + '_features_extra_test_' + str(len(x_test1)) + '_signals.npy', np.array(x_test1))
# x_test2 = learn.get_specific_additional_features(p.sig_test_penny2)
# np.save("test\\penny\\features\\psd_av\\" + 'area_' + str(2) + '_' + "penny" + '_features_extra_test_' + str(len(x_test2)) + '_signals.npy', np.array(x_test2))
# x_test3 = learn.get_specific_additional_features(p.sig_test_penny3)
# np.save("test\\penny\\features\\psd_av\\" + 'area_' + str(3) + '_' + "penny" + '_features_extra_test_' + str(len(x_test3)) + '_signals.npy', np.array(x_test3))
# x_test4 = learn.get_specific_additional_features(p.sig_test_penny4)
# np.save("test\\penny\\features\\psd_av\\" + 'area_' + str(4) + '_' + "penny" + '_features_extra_test_' + str(len(x_test4)) + '_signals.npy', np.array(x_test4))
#
# x_train1 = learn.get_specific_additional_features(p.sig_train_penny1)
# np.save("train\\penny\\features\\psd_av\\" + 'area_' + str(1) + '_' + "penny"+ '_features_extra_train' + str(len(x_train1)) + '_signals.npy', np.array(x_train1))
# x_train2 = learn.get_specific_additional_features(p.sig_train_penny2)
# np.save("train\\penny\\features\\psd_av\\" + 'area_' + str(2) + '_' + "penny"+ '_features_extra_train' + str(len(x_train2)) + '_signals.npy', np.array(x_train2))
# x_train3 = learn.get_specific_additional_features(p.sig_train_penny3)
# np.save("train\\penny\\features\\psd_av\\" + 'area_' + str(3) + '_' + "penny"+ '_features_extra_train' + str(len(x_train3)) + '_signals.npy', np.array(x_train3))
# x_train4 = learn.get_specific_additional_features(p.sig_train_penny4)
# np.save("train\\penny\\features\\psd_av\\" + 'area_' + str(4) + '_' + "penny"+ '_features_extra_train' + str(len(x_train4)) + '_signals.npy', np.array(x_train4))


#############  carmen ########################

# ------------ base features -------------
# x_test1 = learn.get_features(c.sig_test_carmen1)
# np.save("test\\carmen\\features\\" + 'area_' + str(1) + '_' + "carmen" + '_features_extra_test_' + str(len(x_test1)) + '_signals.npy', np.array(x_test1))
# x_test2 = learn.get_features(c.sig_test_carmen2)
# np.save("test\\carmen\\features\\" + 'area_' + str(2) + '_' + "carmen" + '_features_extra_test_' + str(len(x_test2)) + '_signals.npy', np.array(x_test2))
# x_test3 = learn.get_features(c.sig_test_carmen3)
# np.save("test\\carmen\\features\\" + 'area_' + str(3) + '_' + "carmen" + '_features_extra_test_' + str(len(x_test3)) + '_signals.npy', np.array(x_test3))
# x_test4 = learn.get_features(c.sig_test_carmen4)
# np.save("test\\carmen\\features\\" + 'area_' + str(4) + '_' + "carmen" + '_features_extra_test_' + str(len(x_test4)) + '_signals.npy', np.array(x_test4))


# x_train1 = learn.get_features(c.sig_train_carmen1)
# np.save("train\\carmen\\features\\" + 'area_' + str(1) + '_' + "carmen"+ '_features_extra_train' + str(len(x_train1)) + '_signals.npy', np.array(x_train1))
# x_train2 = learn.get_features(c.sig_train_carmen2)
# np.save("train\\carmen\\features\\" + 'area_' + str(2) + '_' + "carmen"+ '_features_extra_train' + str(len(x_train2)) + '_signals.npy', np.array(x_train2))
# x_train3 = learn.get_features(c.sig_train_carmen3)
# np.save("train\\carmen\\features\\" + 'area_' + str(3) + '_' + "carmen"+ '_features_extra_train' + str(len(x_train3)) + '_signals.npy', np.array(x_train3))
# x_train4 = learn.get_features(c.sig_train_carmen4)
# np.save("train\\carmen\\features\\" + 'area_' + str(4) + '_' + "carmen"+ '_features_extra_train' + str(len(x_train4)) + '_signals.npy', np.array(x_train4))

##--------------additional features -----------------------

# x_test1 = learn.get_additional_features(c.sig_test1)
# np.save("test\\carmen\\features\\entropy\\" + 'area_' + str(1) + '_' + "carmen" + '_features_extra_test_' + str(len(x_test1)) + '_signals.npy', np.array(x_test1))
# x_test2 = learn.get_additional_features(c.sig_test2)
# np.save("test\\carmen\\features\\entropy\\" + 'area_' + str(2) + '_' + "carmen" + '_features_extra_test_' + str(len(x_test2)) + '_signals.npy', np.array(x_test2))
# x_test3 = learn.get_additional_features(c.sig_test3)
# np.save("test\\carmen\\features\\entropy\\" + 'area_' + str(3) + '_' + "carmen" + '_features_extra_test_' + str(len(x_test3)) + '_signals.npy', np.array(x_test3))
# x_test4 = learn.get_additional_features(c.sig_test4)
# np.save("test\\carmen\\features\\entropy\\" + 'area_' + str(4) + '_' + "carmen" + '_features_extra_test_' + str(len(x_test4)) + '_signals.npy', np.array(x_test4))

# x_train1 = learn.get_additional_features(c.sig_train1)
# np.save("train\\carmen\\features\\entropy\\" + 'area_' + str(1) + '_' + "carmen"+ '_features_extra_train' + str(len(x_train1)) + '_signals.npy', np.array(x_train1))
# x_train2 = learn.get_additional_features(c.sig_train2)
# np.save("train\\carmen\\features\\entropy\\" + 'area_' + str(2) + '_' + "carmen"+ '_features_extra_train' + str(len(x_train2)) + '_signals.npy', np.array(x_train2))
# x_train3 = learn.get_additional_features(c.sig_train3)
# np.save("train\\carmen\\features\\entropy\\" + 'area_' + str(3) + '_' + "carmen"+ '_features_extra_train' + str(len(x_train3)) + '_signals.npy', np.array(x_train3))
# x_train4 = learn.get_additional_features(c.sig_train4)
# np.save("train\\carmen\\features\\entropy\\" + 'area_' + str(4) + '_' + "carmen"+ '_features_extra_train' + str(len(x_train4)) + '_signals.npy', np.array(x_train4))

# -------------------- psd ------------------------------
 #------------------welch ----------------------
# x_test1 = learn.get_specific_additional_features(c.sig_test_carmen1, "welch")
# np.save("test\\carmen\\features\\psd_welch\\" + 'area_' + str(1) + '_' + "carmen" + '_features_extra_test_' + str(len(x_test1)) + '_signals.npy', np.array(x_test1))
# x_test2 = learn.get_specific_additional_features(c.sig_test_carmen2, "welch")
# np.save("test\\carmen\\features\\psd_welch\\" + 'area_' + str(2) + '_' + "carmen" + '_features_extra_test_' + str(len(x_test2)) + '_signals.npy', np.array(x_test2))
# x_test3 = learn.get_specific_additional_features(c.sig_test_carmen3, "welch")
# np.save("test\\carmen\\features\\psd_welch\\" + 'area_' + str(3) + '_' + "carmen" + '_features_extra_test_' + str(len(x_test3)) + '_signals.npy', np.array(x_test3))
# x_test4 = learn.get_specific_additional_features(c.sig_test_carmen4, "welch")
# np.save("test\\carmen\\features\\psd_welch\\" + 'area_' + str(4) + '_' + "carmen" + '_features_extra_test_' + str(len(x_test4)) + '_signals.npy', np.array(x_test4))

# x_train1 = learn.get_specific_additional_features(c.sig_train_carmen1, "welch")
# np.save("train\\carmen\\features\\psd_welch\\" + 'area_' + str(1) + '_' + "carmen"+ '_features_extra_train' + str(len(x_train1)) + '_signals.npy', np.array(x_train1))
# x_train2 = learn.get_specific_additional_features(c.sig_train_carmen2, "welch")
# np.save("train\\carmen\\features\\psd_welch\\" + 'area_' + str(2) + '_' + "carmen"+ '_features_extra_train' + str(len(x_train2)) + '_signals.npy', np.array(x_train2))
# x_train3 = learn.get_specific_additional_features(c.sig_train_carmen3, "welch")
# np.save("train\\carmen\\features\\psd_welch\\" + 'area_' + str(3) + '_' + "carmen"+ '_features_extra_train' + str(len(x_train3)) + '_signals.npy', np.array(x_train3))
# x_train4 = learn.get_specific_additional_features(c.sig_train_carmen4, "welch")
# np.save("train\\carmen\\features\\psd_welch\\" + 'area_' + str(4) + '_' + "carmen"+ '_features_extra_train' + str(len(x_train4)) + '_signals.npy', np.array(x_train4))

#----------multitaper-------------------
# val = "multitapers"
# x_test1 = learn.get_specific_additional_features(c.sig_test_carmen1, val)
# np.save("test\\carmen\\features\\psd_multitaper\\" + 'area_' + str(1) + '_' + "carmen" + '_features_extra_test_' + str(len(x_test1)) + '_signals.npy', np.array(x_test1))
# x_test2 = learn.get_specific_additional_features(c.sig_test_carmen2, val)
# np.save("test\\carmen\\features\\psd_multitaper\\" + 'area_' + str(2) + '_' + "carmen" + '_features_extra_test_' + str(len(x_test2)) + '_signals.npy', np.array(x_test2))
# x_test3 = learn.get_specific_additional_features(c.sig_test_carmen3, val)
# np.save("test\\carmen\\features\\psd_multitaper\\" + 'area_' + str(3) + '_' + "carmen" + '_features_extra_test_' + str(len(x_test3)) + '_signals.npy', np.array(x_test3))
# x_test4 = learn.get_specific_additional_features(c.sig_test_carmen4, val)
# np.save("test\\carmen\\features\\psd_multitaper\\" + 'area_' + str(4) + '_' + "carmen" + '_features_extra_test_' + str(len(x_test4)) + '_signals.npy', np.array(x_test4))

# x_train1 = learn.get_specific_additional_features(c.sig_train_carmen1,val)
# np.save("train\\carmen\\features\\psd_multitaper\\" + 'area_' + str(1) + '_' + "carmen"+ '_features_extra_train' + str(len(x_train1)) + '_signals.npy', np.array(x_train1))
# x_train2 = learn.get_specific_additional_features(c.sig_train_carmen2, val)
# np.save("train\\carmen\\features\\psd_multitaper\\" + 'area_' + str(2) + '_' + "carmen"+ '_features_extra_train' + str(len(x_train2)) + '_signals.npy', np.array(x_train2))
# x_train3 = learn.get_specific_additional_features(c.sig_train_carmen3, val)
# np.save("train\\carmen\\features\\psd_multitaper\\" + 'area_' + str(3) + '_' + "carmen"+ '_features_extra_train' + str(len(x_train3)) + '_signals.npy', np.array(x_train3))
# x_train4 = learn.get_specific_additional_features(c.sig_train_carmen4, val)
# np.save("train\\carmen\\features\\psd_multitaper\\" + 'area_' + str(4) + '_' + "carmen"+ '_features_extra_train' + str(len(x_train4)) + '_signals.npy', np.array(x_train4))


#-----------burg----------------
# val = "burg"
# x_test1 = learn.get_specific_additional_features(c.sig_test_carmen1, val)
# np.save("test\\carmen\\features\\psd_burg\\" + 'area_' + str(1) + '_' + "carmen" + '_features_extra_test_' + str(len(x_test1)) + '_signals.npy', np.array(x_test1))
# x_test2 = learn.get_specific_additional_features(c.sig_test_carmen2, val)
# np.save("test\\carmen\\features\\psd_burg\\" + 'area_' + str(2) + '_' + "carmen" + '_features_extra_test_' + str(len(x_test2)) + '_signals.npy', np.array(x_test2))
# x_test3 = learn.get_specific_additional_features(c.sig_test_carmen3, val)
# np.save("test\\carmen\\features\\psd_burg\\" + 'area_' + str(3) + '_' + "carmen" + '_features_extra_test_' + str(len(x_test3)) + '_signals.npy', np.array(x_test3))
# x_test4 = learn.get_specific_additional_features(c.sig_test_carmen4, val)
# np.save("test\\carmen\\features\\psd_burg\\" + 'area_' + str(4) + '_' + "carmen" + '_features_extra_test_' + str(len(x_test4)) + '_signals.npy', np.array(x_test4))

# x_train1 = learn.get_specific_additional_features(c.sig_train_carmen1,val)
# np.save("train\\carmen\\features\\psd_burg\\" + 'area_' + str(1) + '_' + "carmen"+ '_features_extra_train' + str(len(x_train1)) + '_signals.npy', np.array(x_train1))
# x_train2 = learn.get_specific_additional_features(c.sig_train2, val)
# np.save("train\\carmen\\features\\psd_burg\\" + 'area_' + str(2) + '_' + "carmen"+ '_features_extra_train' + str(len(x_train2)) + '_signals.npy', np.array(x_train2))
# x_train3 = learn.get_specific_additional_features(c.sig_train_carmen3, val)
# np.save("train\\carmen\\features\\psd_burg\\" + 'area_' + str(3) + '_' + "carmen"+ '_features_extra_train' + str(len(x_train3)) + '_signals.npy', np.array(x_train3))
# x_train4 = learn.get_specific_additional_features(c.sig_train_carmen4, val)
# np.save("train\\carmen\\features\\psd_burg\\" + 'area_' + str(4) + '_' + "carmen"+ '_features_extra_train' + str(len(x_train4)) + '_signals.npy', np.array(x_train4))


#############  menta ########################

# ------------ base features -------------
# x_test1 = learn.get_features(m.sig_test_menta1)
# np.save("test\\menta\\features\\" + 'area_' + str(1) + '_' + "menta" + '_features_extra_test_' + str(
#     len(x_test1)) + '_signals.npy', np.array(x_test1))
# x_test2 = learn.get_features(m.sig_test_menta2)
# np.save("test\\menta\\features\\" + 'area_' + str(2) + '_' + "menta" + '_features_extra_test_' + str(len(x_test2)) + '_signals.npy', np.array(x_test2))
# x_test3 = learn.get_features(m.sig_test_menta3)
# np.save("test\\menta\\features\\" + 'area_' + str(3) + '_' + "menta" + '_features_extra_test_' + str(len(x_test3)) + '_signals.npy', np.array(x_test3))
# x_test4 = learn.get_features(m.sig_test_menta4)
# np.save("test\\menta\\features\\" + 'area_' + str(4) + '_' + "menta" + '_features_extra_test_' + str(len(x_test4)) + '_signals.npy', np.array(x_test4))
#
# x_train1 = learn.get_features(m.sig_train_menta1)
# np.save("train\\menta\\features\\" + 'area_' + str(1) + '_' + "menta"+ '_features_extra_train' + str(len(x_train1)) + '_signals.npy', np.array(x_train1))
# x_train2 = learn.get_features(m.sig_train_menta2)
# np.save("train\\menta\\features\\" + 'area_' + str(2) + '_' + "menta"+ '_features_extra_train' + str(len(x_train2)) + '_signals.npy', np.array(x_train2))
# x_train3 = learn.get_features(m.sig_train_menta3)
# np.save("train\\menta\\features\\" + 'area_' + str(3) + '_' + "menta"+ '_features_extra_train' + str(len(x_train3)) + '_signals.npy', np.array(x_train3))
# x_train4 = learn.get_features(m.sig_train_menta4)
# np.save("train\\menta\\features\\" + 'area_' + str(4) + '_' + "menta" + '_features_extra_train' + str(
#     len(x_train4)) + '_signals.npy', np.array(x_train4))


# -------------------- psd ------------------------------
 #------------------welch ----------------------
# x_test1 = learn.get_specific_additional_features(m.sig_test_menta1, "welch")
# np.save("test\\menta\\features\\psd_welch\\" + 'area_' + str(1) + '_' + "menta" + '_features_extra_test_' + str(len(x_test1)) + '_signals.npy', np.array(x_test1))
# x_test2 = learn.get_specific_additional_features(m.sig_test_menta2, "welch")
# np.save("test\\menta\\features\\psd_welch\\" + 'area_' + str(2) + '_' + "menta" + '_features_extra_test_' + str(len(x_test2)) + '_signals.npy', np.array(x_test2))
# x_test3 = learn.get_specific_additional_features(m.sig_test_menta3, "welch")
# np.save("test\\menta\\features\\psd_welch\\" + 'area_' + str(3) + '_' + "menta" + '_features_extra_test_' + str(len(x_test3)) + '_signals.npy', np.array(x_test3))
# x_test4 = learn.get_specific_additional_features(m.sig_test_menta4, "welch")
# np.save("test\\menta\\features\\psd_welch\\" + 'area_' + str(4) + '_' + "menta" + '_features_extra_test_' + str(len(x_test4)) + '_signals.npy', np.array(x_test4))
#
# x_train1 = learn.get_specific_additional_features(m.sig_train_menta1, "welch")
# np.save("train\\menta\\features\\psd_welch\\" + 'area_' + str(1) + '_' + "menta"+ '_features_extra_train' + str(len(x_train1)) + '_signals.npy', np.array(x_train1))
# x_train2 = learn.get_specific_additional_features(m.sig_train_menta2, "welch")
# np.save("train\\menta\\features\\psd_welch\\" + 'area_' + str(2) + '_' + "menta"+ '_features_extra_train' + str(len(x_train2)) + '_signals.npy', np.array(x_train2))
# x_train3 = learn.get_specific_additional_features(m.sig_train_menta3, "welch")
# np.save("train\\menta\\features\\psd_welch\\" + 'area_' + str(3) + '_' + "menta"+ '_features_extra_train' + str(len(x_train3)) + '_signals.npy', np.array(x_train3))
# x_train4 = learn.get_specific_additional_features(m.sig_train_menta4, "welch")
# np.save("train\\menta\\features\\psd_welch\\" + 'area_' + str(4) + '_' + "menta"+ '_features_extra_train' + str(len(x_train4)) + '_signals.npy', np.array(x_train4))

#----------multitaper-------------------
# val = "multitapers"
# x_test1 = learn.get_specific_additional_features(m.sig_test_menta1, val)
# np.save("test\\menta\\features\\psd_multitaper\\" + 'area_' + str(1) + '_' + "menta" + '_features_extra_test_' + str(len(x_test1)) + '_signals.npy', np.array(x_test1))
# x_test2 = learn.get_specific_additional_features(m.sig_test_menta2, val)
# np.save("test\\menta\\features\\psd_multitaper\\" + 'area_' + str(2) + '_' + "menta" + '_features_extra_test_' + str(len(x_test2)) + '_signals.npy', np.array(x_test2))
# x_test3 = learn.get_specific_additional_features(m.sig_test_menta3, val)
# np.save("test\\menta\\features\\psd_multitaper\\" + 'area_' + str(3) + '_' + "menta" + '_features_extra_test_' + str(len(x_test3)) + '_signals.npy', np.array(x_test3))
# x_test4 = learn.get_specific_additional_features(m.sig_test_menta4, val)
# np.save("test\\menta\\features\\psd_multitaper\\" + 'area_' + str(4) + '_' + "menta" + '_features_extra_test_' + str(len(x_test4)) + '_signals.npy', np.array(x_test4))
#
# x_train1 = learn.get_specific_additional_features(m.sig_train_menta1,val)
# np.save("train\\menta\\features\\psd_multitaper\\" + 'area_' + str(1) + '_' + "menta"+ '_features_extra_train' + str(len(x_train1)) + '_signals.npy', np.array(x_train1))
# x_train2 = learn.get_specific_additional_features(m.sig_train_menta2, val)
# np.save("train\\menta\\features\\psd_multitaper\\" + 'area_' + str(2) + '_' + "menta"+ '_features_extra_train' + str(len(x_train2)) + '_signals.npy', np.array(x_train2))
# x_train3 = learn.get_specific_additional_features(m.sig_train_menta3, val)
# np.save("train\\menta\\features\\psd_multitaper\\" + 'area_' + str(3) + '_' + "menta"+ '_features_extra_train' + str(len(x_train3)) + '_signals.npy', np.array(x_train3))
# x_train4 = learn.get_specific_additional_features(m.sig_train_menta4, val)
# np.save("train\\menta\\features\\psd_multitaper\\" + 'area_' + str(4) + '_' + "menta"+ '_features_extra_train' + str(len(x_train4)) + '_signals.npy', np.array(x_train4))


# x_test1 = learn.get_additional_features(m.sig_test_menta1)
# np.save("test\\menta\\features\\hilbert\\" + 'area_' + str(1) + '_' + "menta" + '_features_extra_test_' + str(len(x_test1)) + '_signals.npy', np.array(x_test1))
# x_test2 = learn.get_additional_features(m.sig_test_menta2)
# np.save("test\\menta\\features\\hilbert\\" + 'area_' + str(2) + '_' + "menta" + '_features_extra_test_' + str(len(x_test2)) + '_signals.npy', np.array(x_test2))
# x_test3 = learn.get_additional_features(m.sig_test_menta3)
# np.save("test\\menta\\features\\hilbert\\" + 'area_' + str(3) + '_' + "menta" + '_features_extra_test_' + str(len(x_test3)) + '_signals.npy', np.array(x_test3))
# x_test4 = learn.get_additional_features(m.sig_test_menta4)
# np.save("test\\menta\\features\\hilbert\\" + 'area_' + str(4) + '_' + "menta" + '_features_extra_test_' + str(len(x_test4)) + '_signals.npy', np.array(x_test4))
#
# x_train1 = learn.get_additional_features(m.sig_train_menta1)
# np.save("train\\menta\\features\\hilbert\\" + 'area_' + str(1) + '_' + "menta"+ '_features_extra_train' + str(len(x_train1)) + '_signals.npy', np.array(x_train1))
# x_train2 = learn.get_additional_features(m.sig_train_menta2)
# np.save("train\\menta\\features\\hilbert\\" + 'area_' + str(2) + '_' + "menta"+ '_features_extra_train' + str(len(x_train2)) + '_signals.npy', np.array(x_train2))
# x_train3 = learn.get_additional_features(m.sig_train_menta3)
# np.save("train\\menta\\features\\hilbert\\" + 'area_' + str(3) + '_' + "menta"+ '_features_extra_train' + str(len(x_train3)) + '_signals.npy', np.array(x_train3))
# x_train4 = learn.get_additional_features(m.sig_train_menta4)
# np.save("train\\menta\\features\\hilbert\\" + 'area_' + str(4) + '_' + "menta"+ '_features_extra_train' + str(len(x_train4)) + '_signals.npy', np.array(x_train4))

# --------------additional features -----------------------

# x_test1 = learn.get_additional_features(m.sig_test_menta1)
# np.save("test\\menta\\features\\additional_featues\\" + 'area_' + str(1) + '_' + "menta" + '_features_extra_test_' + str(
#     len(x_test1)) + '_signals.npy', np.array(x_test1))
# x_test2 = learn.get_additional_features(m.sig_test_menta2)
# np.save("test\\menta\\features\\additional_featues\\" + 'area_' + str(2) + '_' + "menta" + '_features_extra_test_' + str(len(x_test2)) + '_signals.npy', np.array(x_test2))
# x_test3 = learn.get_additional_features(m.sig_test_menta3)
# np.save("test\\menta\\features\\additional_featues\\" + 'area_' + str(3) + '_' + "menta" + '_features_extra_test_' + str(len(x_test3)) + '_signals.npy', np.array(x_test3))
# x_test4 = learn.get_additional_features(m.sig_test_menta4)
# np.save("test\\menta\\features\\additional_featues\\" + 'area_' + str(4) + '_' + "menta" + '_features_extra_test_' + str(len(x_test4)) + '_signals.npy', np.array(x_test4))
# # #
# x_train1 = learn.get_additional_features(m.sig_train_menta1)
# np.save("train\\menta\\features\\additional_featues\\" + 'area_' + str(1) + '_' + "menta"+ '_features_extra_train' + str(len(x_train1)) + '_signals.npy', np.array(x_train1))
# x_train2 = learn.get_additional_features(m.sig_train_menta2)
# np.sve("train\\menta\\features\\additional_featues\\" + 'area_' + str(2) + '_' + "menta"+ '_features_extra_train' + str(len(x_train2)) + '_signals.npy', np.array(x_train2))
# x_train3 = learn.get_additional_features(m.sig_train_menta3)
# np.save("train\\menta\\features\\additional_featues\\" + 'area_' + str(3) + '_' + "menta"+ '_features_extra_train' + str(len(x_train3)) + '_signals.npy', np.array(x_train3))
# x_train4 = learn.get_additional_features(m.sig_train_menta4)
# np.save("train\\menta\\features\\additional_featues\\" + 'area_' + str(4) + '_' + "menta" + '_features_extra_train' + str(
#     len(x_train4)) + '_signals.npy', np.array(x_train4))

# -------------------- psd ------------------------------

# x_test1 = learn.get_specific_additional_features(m.sig_test_menta1)
# np.save("test\\menta\\features\\psd_av\\" + 'area_' + str(1) + '_' + "menta" + '_features_extra_test_' + str(
#     len(x_test1)) + '_signals.npy', np.array(x_test1))
# x_test2 = learn.get_specific_additional_features(m.sig_test_menta2)
# np.save("test\\menta\\features\\psd_av\\" + 'area_' + str(2) + '_' + "menta" + '_features_extra_test_' + str(len(x_test2)) + '_signals.npy', np.array(x_test2))
# x_test3 = learn.get_specific_additional_features(m.sig_test_menta3)
# np.save("test\\menta\\features\\psd_av\\" + 'area_' + str(3) + '_' + "menta" + '_features_extra_test_' + str(len(x_test3)) + '_signals.npy', np.array(x_test3))
# x_test4 = learn.get_specific_additional_features(m.sig_test_menta4)
# np.save("test\\menta\\features\\psd_av\\" + 'area_' + str(4) + '_' + "menta" + '_features_extra_test_' + str(len(x_test4)) + '_signals.npy', np.array(x_test4))

# x_train1 = learn.get_specific_additional_features(m.sig_train_menta1)
# np.save("train\\menta\\features\\psd_av\\" + 'area_' + str(1) + '_' + "menta"+ '_features_extra_train' + str(len(x_train1)) + '_signals.npy', np.array(x_train1))
# x_train2 = learn.get_specific_additional_features(m.sig_train_menta2)
# np.save("train\\menta\\features\\psd_av\\" + 'area_' + str(2) + '_' + "menta"+ '_features_extra_train' + str(len(x_train2)) + '_signals.npy', np.array(x_train2))
# x_train3 = learn.get_specific_additional_features(m.sig_train_menta3)
# np.save("train\\menta\\features\\psd_av\\" + 'area_' + str(3) + '_' + "menta"+ '_features_extra_train' + str(len(x_train3)) + '_signals.npy', np.array(x_train3))
# x_train4 = learn.get_specific_additional_features(m.sig_train_menta4)
# np.save("train\\menta\\features\\psd_av\\" + 'area_' + str(4) + '_' + "menta" + '_features_extra_train' + str(
#     len(x_train4)) + '_signals.npy', np.array(x_train4))
