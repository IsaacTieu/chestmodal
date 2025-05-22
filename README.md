Test Performance by Pathology:
Pathology                 AUC        Accuracy   Precision  Recall     F1
---------------------------------------------------------------------------
No Finding                0.8974     0.8629     0.5667     0.8095     0.6667
Enlarged Cardiomediastinum 0.4792     0.6727     0.0625     0.2500     0.1000
Cardiomegaly              0.6876     0.5948     0.3333     0.8077     0.4719
Lung Opacity              0.7641     0.6833     0.5441     0.8409     0.6607
Lung Lesion               0.3588     0.7742     0.0000     0.0000     0.0000
Edema                     0.5979     0.5630     0.2545     0.5600     0.3500    
Consolidation             0.6389     0.6581     0.1220     0.5556     0.2000
Pneumonia                 0.4746     0.2358     0.1959     0.8636     0.3193
Atelectasis               0.8684     0.6121     0.3919     1.0000     0.5631
Pneumothorax              0.5746     0.7295     0.0370     0.1250     0.0571
Pleural Effusion          0.8739     0.7521     0.5179     0.9062     0.6591    
Pleural Other             0.4918     0.8387     0.0500     0.5000     0.0909
Fracture                  0.5734     0.6774     0.0526     0.3333     0.0909
Support Devices           0.8254     0.7642     0.6316     0.8182     0.7129
---------------------------------------------------------------------------
MEAN                      0.6504     0.6728     0.2686     0.5979     0.3530


docker build -t chestmodal .
docker run -p 8000:8000 chestmodal
