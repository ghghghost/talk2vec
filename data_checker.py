import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt

# 해당 코드는 데이터의 형태 확인을 위해 쓴 간이 코드
# 무시해도 됨.

samplerate, data = scipy.io.wavfile.read('물좀갖다주세요.wav')
times = np.arange(len(data))/float(samplerate)

plt.fill_between(times, data)
plt.xlim(times[0], times[-1])
plt.xlabel('time (s)')
plt.ylabel('amplitude')
plt.show()

print(hi)