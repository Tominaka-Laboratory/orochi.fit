from scipy import interpolate as intp
import numpy as np


class PDFExtensions:
    def __init__(self):
        self["simx_attr"] = "r"
        self["simy_attr"] = "Gr"

    def read_file(self, file_path):
        """
        Reads observed Pair Distribution Function (PDF) data from a specified file.

        This method uses `orochi.utils.file.read_data.Gr` to load PDF data,
        extracts the independent variable (`obsx` as r) and dependent variable (`obsy` as G(r)),
        and assigns them to `self["obsx"]` and `self["obsy"]`.
        Additionally, it attempts to extract metadata from the file (e.g., 'rmin')
        and applies relevant values to the simulator modules if their parameters match.
        Specifically, `rmin` values less than or equal to 0.01 are adjusted to 0.01 to
        prevent potential issues in downstream calculations.

        Parameters
        ----------
        file_path : str
            The path to the PDF data file.

        Notes
        -----
        This method is tailored for reading PDF data formats compatible with `orochi.utils.file.read_data.Gr`.
        It directly updates the optimizer's observed data attributes and potentially
        the simulator's parameters based on file metadata.
        """
        try:
            from orochi.utils.file import read_data
            data, meta = read_data.Gr(file_path)
        except (ImportError, ModuleNotFoundError):
            meta = {}
            pts = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s.startswith("#"):
                        if "=" in s:
                            k, v = s.lstrip("#").split("=", 1)
                            meta[k.strip().lower()] = v.strip()
                    elif s:
                        parts = s.split()
                        if len(parts) >= 2:
                            try:
                                pts.append([float(parts[0]), float(parts[1])])
                            except ValueError:
                                pass
            data = np.array(pts)

        obsx = data[:,0]
        obsy = data[:,1]
        self["obsx"] = obsx
        self["obsy"] = obsy

        for key, value in meta.items():
            for sim in self.simulators:
                if key in sim.par:
                    if key == "rmin":
                        if float(value) <= 0.01:
                            value = 0.01
                    sim[key] = value


    def save_file(self, file_path):
        """
        Save fitted Pair Distribution Function (PDF) data to a specified file.

        Parameters
        ----------
        file_path : str
            The path to the PDF data file.

        """
        import pandas as pd
        import os

        res = {}
        res["obsx"] = self.obsx
        res["obsy"] = self.obsy
        
        for simulator in self.simulators:
            x = getattr(simulator, self["simx_attr"])
            y = getattr(simulator, self["simy_attr"])
            simy_i = intp.interp1d(
                x, y, bounds_error=False, fill_value=0.0
            )(self.obsx)
            res[simulator._name] = simy_i

        data = pd.DataFrame(res)

        ext = os.path.splitext(file_path)[1]
        if len(ext) == 0:
            file_path += ".csv"

        data.to_csv(file_path, index=False)

    @property
    def Rw(self):
        """Returns the weighted profile agreement factor Rw for the current fit."""
        if hasattr(self, "best_scores") and isinstance(self.best_scores, dict):
            if "Rw" in self.best_scores and self.best_scores["Rw"] is not None:
                return float(self.best_scores["Rw"])
        if hasattr(self, "obsy") and self.obsy is not None and len(self.obsy) > 0:
            denom = float(np.sum(np.square(self.obsy)))
            if denom > 0.0:
                return float(np.sqrt(np.sum(np.square(self.obsy - self.simy)) / denom))
        return None
