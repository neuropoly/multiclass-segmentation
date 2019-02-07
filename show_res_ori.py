import torch
import argparse
import warnings

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Path to the network model to use (.pt file).", required=True)
args = parser.parse_args()

with warnings.catch_warnings(): # ignore the potential SourceChangeWarning
    warnings.simplefilter("ignore")
    network = torch.load(args.model, map_location='cpu')

print "Resolution : {}, orientation : {}".format(network.resolution, network.orientation)