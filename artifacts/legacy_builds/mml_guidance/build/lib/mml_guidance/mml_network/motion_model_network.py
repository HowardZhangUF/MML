#!/usr/bin/env python3
""" Motion Model Network from paper:
    Z. I. Bell, R. Sun, K. Volle, P. Ganesh, S. A. Nivison, and W. E.
    Dixon, “Target Tracking Subject to Intermittent Measurements Using
    Attention Deep Neural Networks,” IEEE Control Systems Letters,
    vol. 7, 2023
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple
import math

devicecuda = torch.device("cuda:0")
devicecpu = torch.device("cpu")
torch.set_default_dtype(torch.float32)

Odom = namedtuple("Odom", ("position", "velocity"))

# Attention model from AtLoc: Attention Guided Camera Localization
class AttentionBlock(nn.Module):
    def __init__(self, inputSize=16, outScale=2):
        super().__init__()
        self.inputSize = inputSize
        self.outScale = outScale
        self.gAtt = nn.Linear(inputSize, inputSize // outScale)
        self.thetaAtt = nn.Linear(inputSize, inputSize // outScale)
        self.phiAtt = nn.Linear(inputSize, inputSize // outScale)
        self.alphaAtt = nn.Linear(inputSize // outScale, inputSize)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        xlen = len(x.shape)
        batchSize = 1
        if xlen > 1:
            batchSize = x.size()[0]

        gAtt = self.gAtt(x).view(batchSize, self.inputSize // self.outScale, 1)
        # print("gAtt\n"+str(gAtt))

        thetaAtt = self.thetaAtt(x).view(batchSize, self.inputSize // self.outScale, 1)
        thetaAtt = thetaAtt.permute(0, 2, 1)
        # print("thetaAtt\n"+str(thetaAtt))

        phiAtt = self.phiAtt(x).view(batchSize, self.inputSize // self.outScale, 1)
        # print("phiAtt\n"+str(phiAtt))

        SAtt = torch.matmul(phiAtt, thetaAtt)
        # print("SAtt\n"+str(SAtt))

        SAttC = self.softmax(SAtt)
        # print("SAttC\n"+str(SAttC))

        yAtt = torch.matmul(SAttC, gAtt)
        yAtt = yAtt.view(batchSize, self.inputSize // self.outScale)
        # print("yAtt\n"+str(yAtt))

        alphaAtt = self.alphaAtt(yAtt)
        alphaAtt = torch.squeeze(alphaAtt)
        # print("alphaAtt\n"+str(alphaAtt))
        # print("x\n"+str(x))

        zAtt = alphaAtt + x

        # print("zAtt\n"+str(zAtt))

        return zAtt


class ReplayMemory:
    def __init__(self, memorySize=100, minDistance=0.01):
        self.memorySize = memorySize
        self.memory = []
        self.memoryTensor = []
        self.minDistance = minDistance

    def append(self, *args):
        newOdom = Odom(*args)
        newOdomTensor = torch.cat(newOdom, -1)
        # print(newOdom)
        # print(newOdomTensor)

        # check if new data different enough from old data
        odomIsNew = True
        if len(self.memoryTensor) > 1:
            for odomTensor in self.memoryTensor:
                odomDiff = torch.linalg.norm(newOdomTensor - odomTensor).item()
                # print(odomDiff)
                if odomDiff < self.minDistance:
                    print("ODOM NOT SAVED, TOO CLOSE TO MEMORY")
                    odomIsNew = False
                    break
        if odomIsNew:
            self.memory.append(newOdom)
            self.memoryTensor.append(newOdomTensor)
            print("ODOM SAVED, MEMORY LENGTH " + str(len(self.memory)))
        # print(self.memory)

    def isMemoryFull(self):
        if len(self.memory) >= self.memorySize:
            return True
        else:
            return False

    def sample(self):
        return self.memory

    def clear(self):
        self.memory.clear()
        self.memoryTensor.clear()


class MotionModelBasis(nn.Module):
    def __init__(
        self,
        alpha=0.001,
        numberHiddenLayers=4,
        inputSize=2,
        hiddenSize=64,
        outputSize=16,
        probHiddenDrop=0.1,
        useAttention=False,
    ):
        super().__init__()
        self.inputSize = inputSize

        # network with arbitrary number of hidden layers
        layers = []

        # first add input layer
        layers.append(nn.Linear(self.inputSize, hiddenSize))
        layers.append(nn.ReLU())

        # add hidden layers
        for ii in range(numberHiddenLayers):
            layers.append(nn.Dropout(p=probHiddenDrop))
            layers.append(nn.Linear(hiddenSize, hiddenSize))
            layers.append(nn.ReLU())

        # add output layer
        layers.append(nn.Dropout(p=probHiddenDrop))
        layers.append(nn.Linear(hiddenSize, outputSize))
        layers.append(nn.Tanh())

        # create attention layer
        if useAttention:
            layers.append(AttentionBlock(outputSize))

        # create sequential network
        self.motionModelBasis = nn.Sequential(*layers)

        # move to cuda
        if torch.cuda.is_available():
            self.motionModelBasis = self.motionModelBasis.to(devicecuda)

        self.basisOptimizer = optim.Adam(self.motionModelBasis.parameters(), lr=alpha)

    # implement forward pass
    def forward(self, x):
        xlen = len(x.shape)
        batchSize = 1

        if xlen > 1:
            batchSize = x.size()[0]

        # x = x.to(devicecuda)
        x = x.view(batchSize, self.inputSize)

        Phi = self.motionModelBasis(x)
        Phi = torch.squeeze(Phi)

        # Phi = Phi.to(devicecpu)

        return Phi


class MotionModel:
    def __init__(
        self,
        alpha=0.001,
        numberHiddenLayers=4,
        inputSize=2,
        hiddenSize=64,
        outputSize=16,
        probHiddenDrop=0.1,
        useAttention=False,
        memorySize=100,
        batchSize=25,
        numberEpochs=10,
        minDistance=0.01,
        gamma=0.01,
        k1=0.1,
    ):
        self.PhiNetwork = MotionModelBasis(
            alpha,
            numberHiddenLayers,
            inputSize,
            hiddenSize,
            outputSize,
            probHiddenDrop,
            useAttention,
        )
        self.memorySize = memorySize
        self.batchSize = batchSize
        self.numberEpochs = numberEpochs
        self.replayMemory = ReplayMemory(memorySize, minDistance)
        self.WHat = torch.randn(inputSize, outputSize)
        self.reset()
        if torch.cuda.is_available():
            self.WHat = self.WHat.to(devicecuda)
            self.etaHat = self.etaHat.to(devicecuda)
        self.lastTime = None
        self.gamma = gamma
        self.k1 = k1
        self.useIdx = -1

    def saveModel(self, time, savePath):
        torch.save(
            {
                "time": time,
                "Phi_state_dict": self.PhiNetwork.motionModelBasis.state_dict(),
                "WHat": self.WHat,
            },
            savePath,
        )

    def loadModel(self, loadPath):
        checkpoint = torch.load(loadPath)
        self.PhiNetwork.motionModelBasis.load_state_dict(checkpoint["Phi_state_dict"])
        self.WHat = checkpoint["WHat"]
        # self.lastTime = checkpoint['time']

    def predict(self, time):
        # if first call to learn, initialize
        if self.lastTime is None:
            print("need observation to initialize")
            return

        # calculate time difference then update weight and state estimate
        dt = time - self.lastTime
        self.lastTime = time

        with torch.no_grad():
            # basis
            Phi = self.PhiNetwork(self.etaHat)
            # print("Phi "+str(Phi))

            # prediction
            etaDotHat = torch.matmul(self.WHat, Phi)

            # state estimate derivative
            etaHatDot = etaDotHat
            # print("etaHatDot "+str(etaHatDot))

            # FIX THIS
            # output layer estimate derivative

            self.etaHat += dt * etaHatDot

            # print("self.WHat "+str(self.WHat))
            return self.etaHat.to(devicecpu).numpy(), etaDotHat.to(devicecpu).numpy()

    def learn(self, targetPosxy, targetLinVelxy, time):
        # convert data to tensor and add to memory
        eta = torch.tensor(targetPosxy, dtype=torch.float32)
        etaDot = torch.tensor(targetLinVelxy, dtype=torch.float32)
        self.replayMemory.append(eta, etaDot)

        if torch.cuda.is_available():
            eta = eta.to(devicecuda)
            etaDot = etaDot.to(devicecuda)

        # print("eta "+str(eta))
        # print("etaHat "+str(self.etaHat))
        # print("etaDot "+str(etaDot))

        # if first call to learn, initialize
        if self.lastTime is None:
            self.lastTime = time
            self.etaHat = eta.clone().detach()
            # return

        # calculate time difference then update weight and state estimate
        dt = time - self.lastTime
        self.lastTime = time

        # print("dt "+str(dt))

        # tracking error
        etaTilde = eta - self.etaHat
        # print("etaTilde "+str(etaTilde))

        with torch.no_grad():
            # basis
            Phi = self.PhiNetwork(eta)
            # print("Phi "+str(Phi))

            # prediction
            etaDotHat = torch.matmul(self.WHat, Phi)

            # state estimate derivative
            etaHatDot = etaDotHat + self.k1 * etaTilde
            # etaHatDot = self.k1*etaTilde
            # print("etaHatDot "+str(etaHatDot))

            # FIX THIS
            # output layer estimate derivative

            WHatDot = self.gamma * torch.outer(Phi, etaTilde).t()
            # WHatDot = self.gamma*torch.matmul(Phi,etaTilde)
            # print("WHatDot "+str(WHatDot))

            # print("self.WHat "+str(self.WHat))

            self.etaHat += dt * etaHatDot
            self.WHat += dt * WHatDot

            # print("self.WHat "+str(self.WHat))
            return (
                eta.to(devicecpu).numpy(),
                self.etaHat.to(devicecpu).numpy(),
                etaDot.to(devicecpu).numpy(),
                etaDotHat.to(devicecpu).numpy(),
            )

    def optimize(self):
        if not self.replayMemory.isMemoryFull():
            print(f"MEMORY NOT FILLED. SIZE {len(self.replayMemory.memory)}")
            return -1, False

        self.useIdx += 1
        if self.useIdx % 2 == 0:
            print("USING DATA EVEN")
        else:
            print("NOT USING DATA ODD")
            self.replayMemory.clear()
            return -1, False

        print("MEMORY FILLED OPTIMIZING BASIS")
        self.PhiNetwork.train()

        transitions = self.replayMemory.sample()
        sampleSize = len(transitions)

        # group the transitions into a dict of batch arrays
        batch = Odom(*zip(*transitions))
        # print("batch "+str(batch))

        # get the position and velocity batch, push to cuda
        positionBatch = torch.cat(batch.position)
        velocityBatch = torch.cat(batch.velocity)

        if torch.cuda.is_available():
            positionBatch = positionBatch.to(devicecuda)
            velocityBatch = velocityBatch.to(devicecuda)

        positionBatch = positionBatch.view(sampleSize, -1)
        velocityBatch = velocityBatch.view(sampleSize, -1)

        # print("positionBatch \n"+str(positionBatch))
        # print("velocityBatch \n"+str(velocityBatch))

        # train random batches over number of epochs using MSE
        velocityLosses = []
        for _ in range(self.numberEpochs):
            # generate random set of batches
            batchStart = np.arange(0, sampleSize, self.batchSize, dtype=np.int64)
            indices = np.arange(sampleSize, dtype=np.int64)
            # np.random.shuffle(indices)
            np.random.shuffle(batchStart)
            batches = [indices[ii : ii + self.batchSize] for ii in batchStart]
            # print(batches)

            for batch in batches:
                if len(batch) == self.batchSize:
                    positionBatchii = positionBatch[batch, :]
                    velocityBatchii = velocityBatch[batch, :]
                    # print("positionBatchii \n"+str(positionBatchii))
                    # print("velocityBatchii \n"+str(velocityBatchii))
                    # print("self.WHat \n"+str(self.WHat))

                    velocityHatBatchii = torch.matmul(
                        self.WHat, self.PhiNetwork(positionBatchii).t()
                    ).t()
                    # velocityHatBatchii = velocityHatBatchii.view(self.batchSize,-1)
                    # print("velocityHatBatchii \n"+str(velocityHatBatchii))

                    velocityLoss = (
                        (velocityHatBatchii - velocityBatchii) ** 2.0
                    ).mean()
                    print("velocityLoss " + str(velocityLoss))

                    if math.isnan(velocityLoss.item()):
                        print("VELOCITY LOSS IS NAN, SHUTTINGDOWN")

                    self.PhiNetwork.basisOptimizer.zero_grad()
                    velocityLoss.backward()
                    self.PhiNetwork.basisOptimizer.step()

                    velocityLosses.append(velocityLoss.item())

            batches.clear()

        velocityLossesAvg = np.asarray(velocityLosses).mean().item()
        velocityLosses.clear()
        self.replayMemory.clear()

        return velocityLossesAvg, True

    def reset(self):
        self.etaHat = torch.tensor([0.3, 0], dtype=torch.float32, device=devicecuda)