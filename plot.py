
import re
import matplotlib.pyplot as plt

log_text = """Epoch 10/300
----------
running_loss: {'train': -114.97096529873933, 'val': -116.2828369140625}
Epoch 20/300
----------
running_loss: {'train': -112.9266620982777, 'val': -113.34036636352539}
Epoch 30/300
----------
running_loss: {'train': -107.93957242098723, 'val': -109.12695693969727}
Epoch 40/300
----------
running_loss: {'train': -102.77466167103161, 'val': -104.47949981689453}
Epoch 50/300
----------
running_loss: {'train': -92.06206720525569, 'val': -93.65987777709961}
Saved model checkpoint /tmp/model50.pt
Epoch 60/300
----------
running_loss: {'train': -57.71068434281783, 'val': -55.896568298339844}
Epoch 70/300
----------
running_loss: {'train': -75.06001975319602, 'val': -74.3190689086914}
Epoch 80/300
----------
running_loss: {'train': -60.48279606212269, 'val': -51.31921577453613}
Epoch 90/300
----------
running_loss: {'train': -44.86376224864613, 'val': -47.61846733093262}
Epoch 100/300
----------
running_loss: {'train': -45.96908708052202, 'val': -43.83137893676758}
Saved model checkpoint /tmp/model100.pt
Epoch 110/300
----------
running_loss: {'train': -42.13412232832474, 'val': -36.426063537597656}
Epoch 120/300
----------
running_loss: {'train': -46.26221431385386, 'val': -45.488590240478516}
Epoch 130/300
----------
running_loss: {'train': -43.92236848310991, 'val': -42.458187103271484}
Epoch 140/300
----------
running_loss: {'train': -41.60236532037908, 'val': -43.82830238342285}
Epoch 150/300
----------
running_loss: {'train': -46.4904108914462, 'val': -39.055803298950195}
Saved model checkpoint /tmp/model150.pt
Epoch 160/300
----------
running_loss: {'train': -37.54750650579279, 'val': -38.415510177612305}
Epoch 170/300
----------
running_loss: {'train': -62.58223932439631, 'val': -60.66393280029297}
Epoch 180/300
----------
running_loss: {'train': -42.77992630004883, 'val': -48.586721420288086}
Epoch 190/300
----------
running_loss: {'train': -38.061608401211835, 'val': -40.80799102783203}
Epoch 200/300
----------
running_loss: {'train': -51.36642941561612, 'val': -50.985586166381836}
Saved model checkpoint /tmp/model200.pt
Epoch 210/300
----------
running_loss: {'train': -44.94637298583985, 'val': -42.20169258117676}
Epoch 220/300
----------
running_loss: {'train': -41.89770993319425, 'val': -45.1336669921875}
Epoch 230/300
----------
running_loss: {'train': -70.72403300892225, 'val': -69.38834381103516}
Epoch 240/300
----------
running_loss: {'train': -92.00626650723542, 'val': -90.27708053588867}
Epoch 250/300
----------
running_loss: {'train': -72.35285186767578, 'val': -85.3752212524414}
Saved model checkpoint /tmp/model250.pt
Epoch 260/300
----------
running_loss: {'train': -105.27643099698152, 'val': -103.02295684814453}
Epoch 270/300
----------
running_loss: {'train': -70.52834736217153, 'val': -75.4296989440918}
Epoch 280/300
----------
running_loss: {'train': -74.72598474675958, 'val': -83.53202819824219}
Epoch 290/300
----------
running_loss: {'train': -102.82443376020953, 'val': -103.86247634887695}
Epoch 300/300
----------
running_loss: {'train': -59.47062475031071, 'val': -55.09001541137695}"""

log_text2 = """Epoch 0/300
----------
running_loss: {'train': 0.06560808747731628, 'val': -4.1547814309597015}
Saved model checkpoint /tmp/model0.pt
New best model (val loss=-4.1547814309597015) at epoch 0
Epoch 10/300
----------
running_loss: {'train': 1.7663583728400147, 'val': -1.6649432480335236}
Epoch 20/300
----------
running_loss: {'train': 0.6180156686089255, 'val': 3.2801782488822937}
Epoch 30/300
----------
running_loss: {'train': 1.278105613860217, 'val': -2.67276394367218}
Epoch 40/300
----------
running_loss: {'train': 0.1668122949248011, 'val': -0.24883460998535156}
Epoch 50/300
----------
running_loss: {'train': 0.050674273949963156, 'val': -3.1328604164300486}
Saved model checkpoint /tmp/model50.pt
Epoch 60/300
----------
running_loss: {'train': 0.7149200629104266, 'val': -1.6781886965036392}
Epoch 70/300
----------
running_loss: {'train': 0.49065192653374234, 'val': 0.7695013009943068}
Epoch 80/300
----------
running_loss: {'train': 0.429267109033059, 'val': 0.40112441778182983}
Epoch 90/300
----------
running_loss: {'train': -0.09019660136916427, 'val': 1.8236235734075308}
Epoch 100/300
----------
running_loss: {'train': -0.015531263568184586, 'val': -1.0071228742599487}
Saved model checkpoint /tmp/model100.pt
Epoch 110/300
----------
running_loss: {'train': -0.10981709713285621, 'val': 0.8484239131212234}
Epoch 120/300
----------
running_loss: {'train': 0.04746800119226628, 'val': 2.0067050755023956}
Epoch 130/300
----------
running_loss: {'train': -0.000541167503053476, 'val': 1.2070828080177307}
Epoch 140/300
----------
running_loss: {'train': 0.3328722319142384, 'val': 0.8417288661003113}
Epoch 150/300
----------
running_loss: {'train': 0.1676827648823912, 'val': 1.7874498963356018}
Saved model checkpoint /tmp/model150.pt
Epoch 160/300
----------
running_loss: {'train': -0.08897255767475476, 'val': 1.7251981794834137}
Epoch 170/300
----------
running_loss: {'train': -0.8349479545246472, 'val': -0.21842828392982483}
Epoch 180/300
----------
running_loss: {'train': 0.24483753469857306, 'val': 0.9898810833692551}
Epoch 190/300
----------
running_loss: {'train': 0.10773595896634189, 'val': -1.369838833808899}
Epoch 200/300
----------
running_loss: {'train': -0.19940235931426287, 'val': 0.8921376690268517}
Saved model checkpoint /tmp/model200.pt
Epoch 210/300
----------
running_loss: {'train': 0.7760638770732012, 'val': -0.7034215331077576}
Epoch 220/300
----------
running_loss: {'train': 0.689920276403427, 'val': 1.741388514637947}
Epoch 230/300
----------
running_loss: {'train': 0.507363037629561, 'val': 1.693995475769043}
Epoch 240/300
----------
running_loss: {'train': -0.39158101312138816, 'val': -0.10246235132217407}
Epoch 250/300
----------
running_loss: {'train': 0.3735533586957238, 'val': 1.4704844951629639}
Saved model checkpoint /tmp/model250.pt
Epoch 260/300
----------
running_loss: {'train': 0.2064120772887359, 'val': 0.6078111976385117}
Epoch 270/300
----------
running_loss: {'train': 0.12441147728399787, 'val': 0.6374816000461578}
Epoch 280/300
----------
running_loss: {'train': 0.2935075905936008, 'val': -0.17739873751997948}
Epoch 290/300
----------
running_loss: {'train': -0.5235744481271302, 'val': 2.179424226284027}
Epoch 300/300
----------
"""

# Regex patterns
epoch_pattern = r"Epoch (\d+)/\d+"
loss_pattern = r"running_loss: \{'train': ([0-9.eE+-]+), 'val': ([0-9.eE+-]+)\}"

epochs = []
train_losses = []
val_losses = []

# Extract data
epoch_matches = re.findall(epoch_pattern, log_text2)
loss_matches = re.findall(loss_pattern, log_text2)

for e, (train, val) in zip(epoch_matches, loss_matches):
    epochs.append(int(e))
    train_losses.append(float(train))
    val_losses.append(float(val))

# Plot
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, marker='o', label='Train Loss')
plt.plot(epochs, val_losses, marker='o', label='Validation Loss')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("loss_plot_mus_new.png")


