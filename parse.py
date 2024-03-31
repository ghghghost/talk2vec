import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from datetime import datetime


class Talk2VecDataset(Dataset):
    def __init__(
        self,
        codePath: str,
        pricePath: str,
        inputs: list,
        term: int,
        nelement: int,
        period: (str, str, str),
        cp949=True,
        ngroup: int = None,
    ):

        self.loadFromFile(codePath, pricePath, cp949)
        self.periodInspect(period, term)
        self.inputs = inputs

        #self.samplerate, self.data = scipy.io.wavfile.read('물좀갖다주세요.wav')
        #times = np.arange(len(data)) / float(samplerate)

        self.groups(ngroup)
        self.nelement = nelement

    def __len__(self):
        return self.ngroup

    def __getitem__(self, index):
        msk, self.minTime, self.maxTime = (
            self.groupMsk[index],
            self.groupBegin[index],
            self.groupEnd[index],
        )  # msk should be inversed

        #해당 부분에서 전처리를 진행한 후에 이후에 학습을 위해서 랜덤으로 masking을 진행

        # src[msk] = src[msk] + _src.transpose((0, 2, 1))
        # indices = np.random.permutation(np.arange(self.length, dtype=np.int32))
        # return {
        #     "src": src[indices][: self.nelement],
        #     "index": indices[: self.nelement],
        #     "mask": msk[: self.nelement],
        # }

    def loadFromFile(self, codePath: str, pricePath: str, cp949: bool = True):
        self.rawCode: pd.DataFrame = (
            pd.read_csv(codePath, encoding="CP949") if cp949 else pd.read_csv(codePath)
        )
        #해당 함수를 통해서 File을 load

    def periodInspect(self, period, term):
        self.dateFormat = period[2]
        self.starttime = datetime.strptime(period[0], self.dateFormat)
        self.endtime = datetime.strptime(period[1], self.dateFormat)

        self.termInspect(term)


    #밑에 존재하는 함수는 전처리를 위해서 코드를 짰으나, 모델에 맞게 다시 수정이 필요할 듯함.

    # def termInspect(self, term):
    #     self.term, self.period = term, (self.endtime - self.starttime).days
    #     assert self.term < self.period
    #
    #     groups = self.rawPrice.groupby("tck_iem_cd")
    #     lengths = groups.size()
    #     adjusted_lengths = lengths.reindex(self.stockCode).fillna(0).astype(int).values
    #     valid_codes = adjusted_lengths >= self.term
    #     self.stockCode = self.stockCode[valid_codes].sort_values()
    #     self.rawPrice = (
    #         self.rawPrice[self.rawPrice["tck_iem_cd"].isin(self.stockCode)]
    #         .sort_values(by="tck_iem_cd")
    #         .sort_values(by="Date")
    #     )
    #
    #     groups = self.rawPrice.groupby("tck_iem_cd")
    #
    #     self.infos = np.array(
    #         [
    #             (
    #                 x[0],
    #                 i,
    #                 x[1]["Date"].min(),
    #                 x[1]["Date"].max(),
    #                 len(x[1]),
    #             )
    #             for i, x in enumerate(groups)
    #         ],
    #         dtype=self.stock_info_dtype,
    #     )
    #     self.length = len(self.stockCode)
    #
    # def code2idx(self, code: str):
    #     for i, _code in enumerate(self.stockCode):
    #         if _code == code:
    #             return i
    #     return np.where(self.stockCode.str.match(code))[0]
    #
    # def idx2code(self, idx: int):
    #     return self.stockCode.iloc[idx]
    #
    # def groups(self, ngroup: int):
    #     self.ngroup = ngroup
    #
    #     self.groupMsk = np.random.random_integers(
    #         size=(self.ngroup, self.length), low=0, high=1
    #     ).astype(np.bool_)
    #
    #     startPos = np.random.random_integers(
    #         low=0, high=len(self.timeline) - self.term - 1, size=(self.ngroup)
    #     )
    #     self.groupBegin = self.timeline[startPos]
    #     self.groupEnd = self.timeline[startPos + self.term - 1]
    #     self.groupMsk = (
    #         (
    #             np.tile(self.groupBegin, (len(self.infos["minTime"]), 1)).transpose(
    #                 -1, -2
    #             )
    #             >= np.tile(self.infos["minTime"], (self.ngroup, 1))
    #         )
    #         & (
    #             np.tile(self.groupEnd, (len(self.infos["maxTime"]), 1)).transpose(
    #                 -1, -2
    #             )
    #             <= np.tile(self.infos["maxTime"], (self.ngroup, 1))
    #         )
    #         & self.groupMsk
    #     )
    #
    # def tighten(self, prices: pd.DataFrame, shape) -> pd.DataFrame:
    #     result = (
    #         prices[(prices["Date"] >= self.minTime) & (prices["Date"] <= self.maxTime)]
    #         .reset_index(drop=True)
    #         .sort_values(by="Date")[self.inputs]
    #     )
    #
    #     if result.shape != shape:
    #         result = pd.DataFrame(np.zeros(shape), columns=result.columns.copy())
    #     assert result.shape == shape
    #     return result

    def ncodes(self):
        return self.length
