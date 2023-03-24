import subprocess
import numpy as np
import pytest
import os
import csv
from dataclasses import dataclass

outputfile = os.path.join(os.getcwd(), 'tests', 'test_conversions.csv')
truth_dir = os.path.join(os.getcwd(), 'tests', 'true_test_results')


@dataclass
class ConversionsOutput:
    ecoliCoreOutput = None
    ecoliCoreReturnCode = None
    conversions = None
    metabIds = None

    def __init__(self, commandList):
        # Run ecmtool on e coli core network
        if os.path.isfile(outputfile):
            os.remove(outputfile)
        self.ecoliCoreOutput = subprocess.run(commandList, stdout=subprocess.PIPE, text=True)

        self.ecoliCoreReturnCode = self.ecoliCoreOutput.returncode
        with open(outputfile, newline='') as f:
            reader = csv.reader(f)
            self.metabIds = next(reader)
        self.conversions = np.loadtxt(outputfile, delimiter=',', skiprows=1)

    def normalize_conversions(self, true_metabIds):
        # Order metabolites in the same way for newly found conversions
        trueMetabIndices = np.array([true_metabIds.index(metab) for metab in self.metabIds])
        conversions_loc = self.conversions.copy()
        conversions_loc[:, trueMetabIndices]

        # Normalise conversions such that all conversions sum up to 1
        conversions_loc /= np.sum(conversions_loc, axis=1, keepdims=True)

        # Order conversions based on first metabolite, then on second, etc,...
        # argsort = np.lexsort(tuple(map(tuple, conversions_loc.T)))
        # conversions_loc = conversions_loc[argsort, :]
        return conversions_loc


def get_metabs(filename):
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        true_metabIds = next(reader)
    return true_metabIds


def areConversionSetsEqual(conversions1, conversions2, atol=1e-8, rtol=1e-5):
    ub = atol + rtol * np.abs(conversions2)
    nMetabs = conversions2.shape[1]
    for conv in conversions1:
        closenessTF = np.abs(conversions2 - conv) <= ub
        indsMatch = np.where(np.sum(closenessTF, axis=1) == nMetabs)[0]
        if len(indsMatch) != 1:
            return False
        else:
            conversions2 = np.delete(conversions2, indsMatch[0], 0)
            ub = np.delete(ub, indsMatch[0], 0)
    return True


class TestEcolicore:
    @pytest.fixture(scope='class', autouse=True)
    def runEcolicore(self):
        # global metabIds, conversions, ecoliCoreReturnCode, outputfile, ecoliCoreOutput
        commandList = ['python', 'main.py', '--model_path', 'models/e_coli_core.xml', '--auto_direction', 'true',
                       '--out_path', outputfile]
        ecoliOutput = ConversionsOutput(commandList)
        return ecoliOutput

    def test_ecolicoreRunCompleted(self, runEcolicore):
        assert (runEcolicore.ecoliCoreReturnCode == 0)

    def test_numberECMs(self, runEcolicore):
        assert (runEcolicore.conversions.shape[0] == 689)

    def test_numberMetabs(self, runEcolicore):
        assert (len(runEcolicore.metabIds) == 21)

    def test_metabIds(self, runEcolicore):
        trueFilename = os.path.join(truth_dir, 'true_ecolicore_conversions.csv')
        true_metabIds = get_metabs(trueFilename)
        assert (sorted(true_metabIds) == sorted(runEcolicore.metabIds))

    def test_conversions(self, runEcolicore):
        trueFilename = os.path.join(truth_dir, 'true_ecolicore_conversions.csv')
        true_metabIds = get_metabs(trueFilename)

        # These true conversions are normalised such that each row sums to 1.
        true_conversions = np.loadtxt(trueFilename, delimiter=',', skiprows=1)
        # Do the same for the current conversions
        conversions_loc = runEcolicore.normalize_conversions(true_metabIds)

        assert areConversionSetsEqual(conversions_loc, true_conversions)


class TestEcolicoreDirect:
    @pytest.fixture(scope='class', autouse=True)
    def runEcolicore(self):
        # global metabIds, conversions, ecoliCoreReturnCode, outputfile, ecoliCoreOutput
        commandList = ['python', 'main.py', '--model_path', 'models/e_coli_core.xml', '--auto_direction', 'true',
                       '--out_path', outputfile, '--direct', 'True']
        ecoliOutput = ConversionsOutput(commandList)
        return ecoliOutput

    def test_ecolicoreRunCompleted(self, runEcolicore):
        assert (runEcolicore.ecoliCoreReturnCode == 0)

    def test_numberECMs(self, runEcolicore):
        assert (runEcolicore.conversions.shape[0] == 689)

    def test_numberMetabs(self, runEcolicore):
        assert (len(runEcolicore.metabIds) == 21)

    def test_metabIds(self, runEcolicore):
        trueFilename = os.path.join(truth_dir, 'true_ecolicore_conversions.csv')
        true_metabIds = get_metabs(trueFilename)
        assert (sorted(true_metabIds) == sorted(runEcolicore.metabIds))

    def test_conversions(self, runEcolicore):
        trueFilename = os.path.join(truth_dir, 'true_ecolicore_conversions.csv')
        true_metabIds = get_metabs(trueFilename)

        # These true conversions are normalised such that each row sums to 1.
        true_conversions = np.loadtxt(trueFilename, delimiter=',', skiprows=1)
        # Do the same for the current conversions
        conversions_loc = runEcolicore.normalize_conversions(true_metabIds)

        assert (areConversionSetsEqual(conversions_loc, true_conversions))
