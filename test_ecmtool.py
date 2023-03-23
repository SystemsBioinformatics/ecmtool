import subprocess
import numpy as np
import pytest
import os
import csv
from dataclasses import dataclass


@dataclass
class ConversionsOutput:
    outputfile = 'test_conversions.csv'
    ecoliCoreOutput = None
    ecoliCoreReturnCode = None
    conversions = None
    metabIds = None


class TestEcolicore:
    @pytest.fixture(scope='class', autouse=True)
    def runEcolicore(self):
        # global metabIds, conversions, ecoliCoreReturnCode, outputfile, ecoliCoreOutput
        # Run ecmtool on e coli core network
        ecoliOutput = ConversionsOutput()
        if os.path.isfile(ecoliOutput.outputfile):
            os.remove(ecoliOutput.outputfile)
        commandList = ['python', 'main.py', '--model_path', 'models/e_coli_core.xml', '--auto_direction', 'true',
                       '--out_path', ecoliOutput.outputfile]
        ecoliOutput.ecoliCoreOutput = subprocess.run(commandList, stdout=subprocess.PIPE, text=True)

        ecoliOutput.ecoliCoreReturnCode = ecoliOutput.ecoliCoreOutput.returncode
        with open(ecoliOutput.outputfile, newline='') as f:
            reader = csv.reader(f)
            ecoliOutput.metabIds = next(reader)
        ecoliOutput.conversions = np.loadtxt(ecoliOutput.outputfile, delimiter=',', skiprows=1)
        return ecoliOutput

    def test_ecolicoreRunCompleted(self, runEcolicore):
        assert (runEcolicore.ecoliCoreReturnCode == 0)

    def test_numberECMs(self, runEcolicore):
        assert (runEcolicore.conversions.shape[0] == 689)

    def test_numberMetabs(self, runEcolicore):
        assert (len(runEcolicore.metabIds) == 21)

    def test_metabIds(self, runEcolicore):
        trueFilename = os.path.join(os.getcwd(), 'true_test_results', 'true_ecolicore_conversions.csv')
        with open(trueFilename, newline='') as f:
            reader = csv.reader(f)
            true_metabIds = next(reader)
        assert (sorted(true_metabIds) == sorted(runEcolicore.metabIds))

    def test_conversions(self, runEcolicore):
        trueFilename = os.path.join(os.getcwd(), 'true_test_results', 'true_ecolicore_conversions.csv')
        with open(trueFilename, newline='') as f:
            reader = csv.reader(f)
            true_metabIds = next(reader)

        # These true conversions are normalised such that each row sums to 1.
        true_conversions = np.loadtxt(trueFilename, delimiter=',', skiprows=1)

        # Order metabolites in the same way for newly found conversions
        trueMetabIndices = np.array([true_metabIds.index(metab) for metab in runEcolicore.metabIds])
        conversions_loc = runEcolicore.conversions.copy()
        conversions_loc[:, trueMetabIndices]

        # Normalise conversions such that all conversions sum up to 1
        conversions_loc /= np.sum(conversions_loc, axis=1, keepdims=True)

        # Order conversions based on first metabolite, then on second, etc,...
        argsort = np.lexsort(tuple(map(tuple, conversions_loc.T)))
        conversions_loc = conversions_loc[argsort, :]

        assert (np.allclose(conversions_loc, true_conversions, rtol=1e-05, atol=1e-08))
