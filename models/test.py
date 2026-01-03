
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
ages_x = [300, 400, 500, 600, 700, 800,900]
dev_1 = [50.46, 52.16, 52.69, 53.13, 53.06, 52.87, 52.88]
plt.plot(ages_x, dev_1, label="MRR", color="#FF0000", marker=".", linestyle="-")
MRR = [39.14, 40.21, 40.55, 41.60, 41.06, 41.13, 41.13]
plt.plot(ages_x, MRR, label="H@1", color="#00FF00", marker=".", linestyle="-")
dev_3 = [58.25, 60.18, 60.39, 60.72, 60.65, 60.77, 60.17]
plt.plot(ages_x, dev_3, label="H@3", color="#6495ED", marker=".", linestyle="-")
plt.xlabel("Embedding dimension")
plt.ylabel("Value(%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.ylim((37, 62))
my_y_ticks = np.arange(37, 62, 3)
plt.yticks(my_y_ticks)
# plt.gcf().set_figheight(4.4)
# plt.gcf().set_figwidth(7)
# plt.savefig("Embedding_dimension.pdf")
plt.show()
