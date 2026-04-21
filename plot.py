
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

log_text2 = """
Epoch 0/300
----------
running_loss: {'train': 1.1073680303313513, 'val': 0.9746958315372467}
Saved model checkpoint /tmp/model0.pt
New best model (val loss=0.9746958315372467) at epoch 0
Epoch 10/300
----------
running_loss: {'train': 0.6356151429089634, 'val': 0.6135304272174835}
Epoch 20/300
----------
running_loss: {'train': 0.6044063622301273, 'val': 0.5806695222854614}
Epoch 30/300
----------
running_loss: {'train': 0.565069550817663, 'val': 0.563374787569046}
Epoch 40/300
----------
running_loss: {'train': 0.5462134886871685, 'val': 0.5562227964401245}
Epoch 50/300
----------
running_loss: {'train': 0.5428041382269427, 'val': 0.5422726273536682}
Saved model checkpoint /tmp/model50.pt
New best model (val loss=0.5422726273536682) at epoch 50
Epoch 60/300
----------
running_loss: {'train': 0.5338907512751492, 'val': 0.5486455261707306}
Epoch 70/300
----------
running_loss: {'train': 0.5338320461186494, 'val': 0.5375697612762451}
Epoch 80/300
----------
running_loss: {'train': 0.5278563391078602, 'val': 0.5370552241802216}
Epoch 90/300
----------
running_loss: {'train': 0.5437717762860386, 'val': 0.5615746676921844}
Epoch 100/300
----------
running_loss: {'train': 0.5234545550563119, 'val': 0.5300788283348083}
Saved model checkpoint /tmp/model100.pt
New best model (val loss=0.5300788283348083) at epoch 100
Epoch 110/300
----------
running_loss: {'train': 0.5230836814100093, 'val': 0.5239729285240173}
Epoch 120/300
----------
running_loss: {'train': 0.5225043947046454, 'val': 0.5242983400821686}
Epoch 130/300
----------
running_loss: {'train': 0.5159729692068966, 'val': 0.5211068242788315}
Epoch 140/300
----------
running_loss: {'train': 0.5146747014739297, 'val': 0.5209836810827255}
Epoch 150/300
----------
running_loss: {'train': 0.5183245675130325, 'val': 0.527680516242981}
Saved model checkpoint /tmp/model150.pt
New best model (val loss=0.527680516242981) at epoch 150
Epoch 160/300
----------
running_loss: {'train': 0.5262904952872883, 'val': 0.5591558516025543}
Epoch 170/300
----------
running_loss: {'train': 0.5249419862573798, 'val': 0.5228519290685654}
Epoch 180/300
----------
running_loss: {'train': 0.5942170511592518, 'val': 0.5771166682243347}
Epoch 190/300
----------
running_loss: {'train': 0.5503397150473162, 'val': 0.5835314095020294}
Epoch 200/300
----------
running_loss: {'train': 0.5276886983351273, 'val': 0.5503818392753601}
Saved model checkpoint /tmp/model200.pt
Epoch 210/300
----------
running_loss: {'train': 0.5214472873644396, 'val': 0.5469220876693726}
Epoch 220/300
----------
running_loss: {'train': 0.513810174031691, 'val': 0.5353141129016876}
Epoch 230/300
----------
running_loss: {'train': 0.5084676769646731, 'val': 0.5327062606811523}
Epoch 240/300
----------
running_loss: {'train': 0.5019866309382699, 'val': 0.5287696570158005}
Epoch 250/300
----------
running_loss: {'train': 0.5049501115625554, 'val': 0.5374244898557663}
Saved model checkpoint /tmp/model250.pt
Epoch 260/300
----------
running_loss: {'train': 0.49906993725083093, 'val': 0.5405613481998444}
Epoch 270/300
----------
running_loss: {'train': 0.49380003864114935, 'val': 0.5049603134393692}
Epoch 280/300
----------
running_loss: {'train': 0.5032140233299949, 'val': 0.5255872160196304}
Epoch 290/300
----------
running_loss: {'train': 0.4960990927436135, 'val': 0.5281639844179153}
Epoch 300/300
----------
running_loss: {'train': 0.48892349546605895, 'val': 0.5247691720724106}
"""

# Regex patterns
epoch_pattern = r"Epoch (\d+)/\d+"
loss_pattern = r"running_loss: \{'train': ([0-9.eE+-]+), 'val': ([0-9.eE+-]+)\}"

epochs = []
train_losses = []
val_losses = []

# Extract data
epoch_matches = re.findall(epoch_pattern, log_text)
loss_matches = re.findall(loss_pattern, log_text)

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
plt.savefig("loss_plot_mus.png")


