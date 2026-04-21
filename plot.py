
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
running_loss: {'train': 0.782041383451239, 'val': 3.7204291820526123}
Saved model checkpoint /tmp/model0.pt
New best model (val loss=3.7204291820526123) at epoch 0
Epoch 10/300
----------
running_loss: {'train': 0.49510333438714355, 'val': 0.3200167417526245}
Epoch 20/300
----------
running_loss: {'train': -1.7511090276141963, 'val': 1.4330227375030518}
Epoch 30/300
----------
running_loss: {'train': -0.1841141631205877, 'val': -4.036914825439453}
Epoch 40/300
----------
running_loss: {'train': -0.7619918187459309, 'val': 2.8895671367645264}
Epoch 50/300
----------
running_loss: {'train': 0.253321409225464, 'val': -3.0956978797912598}
Saved model checkpoint /tmp/model50.pt
New best model (val loss=-3.0956978797912598) at epoch 50
Epoch 60/300
----------
running_loss: {'train': 0.9138693660497667, 'val': -0.8779115676879883}
Epoch 70/300
----------
running_loss: {'train': -0.4441170990467071, 'val': 0.8460376858711243}
Epoch 80/300
----------
running_loss: {'train': -0.6430756201346716, 'val': 0.32579776644706726}
Epoch 90/300
----------
running_loss: {'train': 1.7186450213193893, 'val': -4.296220302581787}
Epoch 100/300
----------
running_loss: {'train': 0.41994398832321167, 'val': 1.1189132928848267}
Saved model checkpoint /tmp/model100.pt
Epoch 110/300
----------
running_loss: {'train': -1.1502921183904011, 'val': 5.154290676116943}
Epoch 120/300
----------
running_loss: {'train': 0.14895632863044742, 'val': 0.6694649457931519}
Epoch 130/300
----------
running_loss: {'train': -0.32035605112711585, 'val': -2.99161434173584}
Epoch 140/300
----------
running_loss: {'train': 0.4012532035509746, 'val': -2.0120012760162354}
Epoch 150/300
----------
running_loss: {'train': 0.13671496510505665, 'val': -1.0986762046813965}
Saved model checkpoint /tmp/model150.pt
Epoch 160/300
----------
running_loss: {'train': 0.6615686444565653, 'val': -0.01335157174617052}
Epoch 170/300
----------
running_loss: {'train': 0.3068748911221822, 'val': -1.4133868217468262}
Epoch 180/300
----------
running_loss: {'train': -0.09218523403008777, 'val': -1.2314435243606567}
Epoch 190/300
----------
running_loss: {'train': 1.0422358413537343, 'val': 2.046855926513672}
Epoch 200/300
----------
running_loss: {'train': 0.593264639377594, 'val': 0.27409058809280396}
Saved model checkpoint /tmp/model200.pt
Epoch 210/300
----------
running_loss: {'train': -0.4587003129223982, 'val': -0.052213262766599655}
Epoch 220/300
----------
running_loss: {'train': 0.6071868687868119, 'val': -1.35789155960083}
Epoch 230/300
----------
running_loss: {'train': 0.4952979485193889, 'val': 2.225036144256592}
Epoch 240/300
----------
running_loss: {'train': -0.9106202945113182, 'val': -1.1720166206359863}
Epoch 250/300
----------
running_loss: {'train': 0.26764077444871265, 'val': 1.7799195051193237}
Saved model checkpoint /tmp/model250.pt
Epoch 260/300
----------
running_loss: {'train': 0.5332959219813347, 'val': -0.809014618396759}
Epoch 270/300
----------
running_loss: {'train': -0.14920836190382636, 'val': -1.541227102279663}
Epoch 280/300
----------
running_loss: {'train': 0.3082664894560973, 'val': 1.0884795188903809}
Epoch 290/300
----------
running_loss: {'train': -0.8198351934552193, 'val': 0.3191615045070648}
Epoch 300/300
----------
running_loss: {'train': 0.3416551550229391, 'val': -1.9507384300231934}
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


