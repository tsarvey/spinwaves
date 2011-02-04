"""
Disclaimer
==========

This software was developed at the National Institute of Standards and Technology at the NIST Center for Neutron Research by employees of the Federal Government in the course of their official duties. Pursuant to title 17 section 105* of the United States Code this software is not subject to copyright protection and is in the public domain. The SPINAL software package is an experimental spinwave analysis system. NIST assumes no responsibility whatsoever for its use, and makes no guarantees, expressed or implied, about its quality, reliability, or any other characteristic. The use of certain trade names or commercial products does not imply any endorsement of a particular product, nor does it imply that the named product is necessarily the best product for the stated purpose. We would appreciate acknowledgment if the software is used.

*Subject matter of copyright: United States Government works

Copyright protection under this title is not available for any work of the United States Government, but the United States Government is not precluded from receiving and holding copyrights transferred to it by assignment, bequest, or otherwise."""


import os
import signal
from multiprocessing import Process, Queue
from threading import Thread
from copy import deepcopy
import wx
#from wx.py.dispatcher import send
import numpy as np
import matplotlib.pyplot as plt
import spinwaves.spinwavecalc.spinwave_calc_file as spinwave_calc_file
import spinwaves.cross_section.util.printing as printing
from spinwaves.MonteCarlo.CSim import ShowSimulationFrame
from spinwaves.vtkModel.wxGUI.Session import Session
from spinwaves.cross_section.csection_calc import cs_driver, load_plot
from spinwaves.utilities.fitting import showFitResultFrame, fitFromFile, annealFitFromFile, showParamListFrame
from spinwaves.spinwavecalc.spinwavepanel import showEditorWindow

tmpDir = os.path.join(os.path.split(os.path.split(os.path.dirname(__file__))[0])[0], "spinwaves_temp")
tmpDir = os.path.join(os.getcwd(), "spinwaves_temp")#py2exe
if not os.path.exists(tmpDir):
    os.mkdir(tmpDir)
print "temp directory: ", tmpDir

def createFileCopy(fileName, pid, type):
    """Creates a copy of the file in tmpDir with a unique name using the pid and returns the new path.
    type is 0 for an interaction file and 1 for a spin file.
    The file name is of the form:
    int_pid   ...or...   spin_pid
    By having a predictable file name, the files can easilly found outside of the process that created them."""
    #newPath = os.path.join(tmpDir, os.path.split(fileName)[1] + "_" + str(pid))
    newPath = tmpFileName(pid, type)
    f2 = open(newPath, 'w')
    f1 = open(fileName, 'r')
    f2.write(f1.read())
    f1.close()
    f2.close()
    return newPath

def tmpFileName(pid, type):
    """type is 0 for an interaction file, 1 for a spin file, and 2 for fitting Record.
    The file name is of the form:
    int_pid.txt   ...or...   spin_pid.txt   ...or...   fitRecord_pid.txt
    By having a predictable file name, the files can easilly found outside of the process that created them."""
    if type==0:
        newName = "int_" + str(pid) + ".txt"
    if type==1:
        newName = "spin_" + str(pid) + ".txt"
    if type==2:
        newName = "fitRecord_" + str(pid) + ".txt"
    return os.path.join(tmpDir, newName)

def createView(processManager):
    """Will display and return a frame object containing a ProcessManagerPanel
    object which is connected to processManager."""
    #Create a View
    frame = wx.Frame(processManager.parent, -1, title = "Processes")
    processManager.setView(ProcessManagerPanel(processManager, frame, -1))
    frame.Fit()
    frame.SetMinSize(frame.GetSize())
    frame.Show()
    return frame

class ProcessManager():
    def __init__(self, parentWindow = None):
        """All results will be displayed in windows which are direct children of parentWindow.  This should probably be the main GUI Frame."""
        self.parent = parentWindow
        #Threads to poll the process Queues
        self._analyticDispThread = None
        self._numericDispThread = None
        self._fitThread = None
        self._crossSecThread = None
        self._calcHsaveThread = None
        
        self._analyticDispQueue = Queue()
        self._numericDispQueue = Queue()
        self._fitQueue = Queue()
        self._crossSecQueue = Queue()
        self._calcHsaveQueue = Queue()
        
        self._analyticDispProcesses= []
        self._numericDispProcesses= []
        self._fitProcesses= []
        self._crossSecProcesses= []
        self._calcHsaveProcesses = []
        #self.processes = []
        
        #Storing snapshots of the current fitting state
        self._fitSnapshots = {}#pid to param list mapping
        self._fitInfoThread = None
        self._fitRecordQueue = Queue()
        
    def setView(self, procManagerPanel):
        self.view = procManagerPanel
    
    def processDone(self, pid):
        """This method should be called by the thread receiving the data from the process when data is received,
        i.e. when the process is done executing."""
        self.view.removeProcess(pid)
        for i in range(len(self._analyticDispProcesses)):
            if self._analyticDispProcesses[i].pid == pid:
                del self._analyticDispProcesses[i]
                if self._analyticDispQueue.empty():
                    self._analyticDispThread = None
                return
        for i in range(len(self._numericDispProcesses)):
            if self._numericDispProcesses[i].pid == pid:
                del self._numericDispProcesses[i]
                if self._numericDispQueue.empty():
                    self._numericDispThread = None
                return
        for i in range(len(self._crossSecProcesses)):
            if self._crossSecProcesses[i].pid == pid:
                del self._crossSecProcesses[i]
                if self._crossSecQueue.empty():
                    self._crossSecThread = None
                return
        for i in range(len(self._calcHsaveProcesses)):
            if self._calcHsaveProcesses[i].pid == pid:
                del self._calcHsaveProcesses[i]
                if self._calcHsaveQueue.empty():
                    self._calcHsaveThread = None
                return
        for i in range(len(self._fitProcesses)):
            if self._fitProcesses[i].pid == pid:
                del self._fitProcesses[i]
                del self._fitSnapshots[pid]
                return
        
    def killProcess(self, pid):
        """Currently the terminate() method is called which runs the risk of corrupting a queue if it is being accessed
        when the process is killed."""
        for p in self._analyticDispProcesses:
            if p.pid == pid:
                p.terminate()
                return
        for p in self._numericDispProcesses:
            if p.pid == pid:
                p.terminate()
                return
        for p in self._fitProcesses:
            if p.pid == pid:
                p.terminate()
                return
        for p in self._crossSecProcesses:
            if p.pid == pid:
                p.terminate()
                return
        for p in self._calcHsaveProcesses:
            if p.pid == pid:
                p.terminate()
                return
    
    def updateFit(self, pid, data):
        self._fitSnapshots[pid] = data
    
    def getProcessInfo(self, pid):
        """Returns:
        process_type, info
        
        processType is one of the following strings: 1)AnalyticDisp 2)NumericDisp 3)Fit 4)AnalyticCrossSec
        
        if process_type is 'AnalyticDisp', NumericDisp', or 'AnalyticCrossSec', info will be a tuple of:
        (interaction_file, spin_file)."""
        processType = ""
        for p in self._analyticDispProcesses:
            if p.pid == pid:
                processType = "AnalyticDisp"
                return processType, (tmpFileName(pid, 0), tmpFileName(pid, 1))
        for p in self._numericDispProcesses:
            if p.pid == pid:
                processType = "NumericDisp"
                return processType, (tmpFileName(pid, 0), tmpFileName(pid, 1))
        for p in self._fitProcesses:
            if p.pid == pid:
                processType = "Fit"
                return processType, self._fitSnapshots[pid]
        for p in self._crossSecProcesses:
            if p.pid == pid:
                processType = "AnalyticCrossSec"
                return processType, (tmpFileName(pid, 0), tmpFileName(pid, 1))
        for p in self._calcHsaveProcesses:
            if p.pid == pid:
                processType = "CalcHsave"
                return processType, (tmpFileName(pid, 0), tmpFileName(pid, 1))
        
    def startAnalyticDispersion(self, Hsave):
        p = Process(target=AnalyticDispFunc, args=(self._analyticDispQueue, Hsave))
        self._analyticDispProcesses.append(p)
        #self.processes.append(p)
        p.start()
        if self.view:
            self.view.AddProcess(p.pid, "Analytic Dispersion", "running")
        #AnalyticDispFunc(self._analyticDispQueue, interaction_file, spin_file)
        if self._analyticDispThread == None:
            self._analyticDispThread = AnalyticDispersionThread(self.parent, self._analyticDispQueue, self)
            self._analyticDispThread.start()
       
    
    def startNumericDispersion(self, Hsave, direction, k_min, k_max, steps): 
        p = Process(target = NumericDispFunc, args = (self._numericDispQueue, Hsave, direction, k_min, k_max, steps))
        self._numericDispProcesses.append(p)
        #self.processes.append(p)
        p.start()
        if self.view:
            self.view.AddProcess(p.pid, "Numerical Dispersion", "running")
        #NumericDispFunc(self._numericDispQueue, interaction_file, spin_file, direction, k_min, k_max, steps)
        if self._numericDispThread == None:
            self._numericDispThread = NumericDispersionThread(self.parent, self._numericDispQueue, self)
            self._numericDispThread.start()

    
    def startFit(self, session, fileName, k, tmin, tmax, size, tfactor, useMC, fitType = 0):
        """if fitType is 0, mp_fit is used, if it is 1, simulated annealing is used."""
        sessXML = session.createXMLStr()
        if fitType == 0:
            p = Process(target = FitFunc, args = (self._fitQueue, sessXML, fileName, k, tmin, tmax, size, tfactor, useMC, self._fitRecordQueue))
        if fitType == 1:
            #AnnealFitFunc(self._fitQueue, sessXML, fileName, k, tmin, tmax, size, tfactor, useMC)
            p = Process(target = AnnealFitFunc, args = (self._fitQueue, sessXML, fileName, k, tmin, tmax, size, tfactor, useMC, self._fitRecordQueue))
        self._fitProcesses.append(p)
        #self.processes.append(p)
        p.start()
        if self.view:
            self.view.AddProcess(p.pid, "Fitting", "running")
        if self._fitThread == None:
            self._fitThread = FitThread(self.parent, self._fitQueue, self)
            self._fitThread.start()
        if self._fitInfoThread == None:
            self._fitInfoThread = FitSnapshotThread(self._fitRecordQueue, self)
            self._fitInfoThread.start()

    def startCrossSection(self, interaction_file, spin_file, hkl_interval, w_interval, tau_list, direction,
                                  temperature, outpath, sphavg_bool, plotchars):
        #AnalyticCrossSectionFunc(self._analyticCrossSecQueue, interaction_file, spin_file)
        args = (interaction_file, spin_file, hkl_interval, w_interval, tau_list, direction,
                temperature, outpath, sphavg_bool, plotchars)
        p = Process(target = CrossSectionFunc, args = (self._crossSecQueue, args))
        self._crossSecProcesses.append(p)
        #self.processes.append(p)
        p.start()
        if self.view:
            self.view.AddProcess(p.pid, "Cross Section Calculation", "running")
        if self._crossSecThread == None:
            self._crossSecThread = CrossSectionThread(self.parent, self._crossSecQueue, self)
            self._crossSecThread.start()
            
    def startCalcHsave(self, interaction_file, spin_file, direction, k_min, k_max, steps):
        p = Process(target=CalcHsaveFunc, args=(self._calcHsaveQueue, interaction_file, spin_file, direction, k_min, k_max, steps))
        self._calcHsaveProcesses.append(p)
        #self.processes.append(p)
        p.start()
        if self.view:
            self.view.AddProcess(p.pid, "Calculating Hsave", "running")
        #AnalyticDispFunc(self._analyticDispQueue, interaction_file, spin_file)
        if self._calcHsaveThread == None:
            self._calcHsaveThread = CalcHsaveThread(self.parent, self._calcHsaveQueue, self)
            self._calcHsaveThread.start()




class ProcessManagerPanel(wx.Panel):
    def __init__(self, procManager, *args, **kwds):
        # begin wxGlade: ProcessManagerPanel.__init__
        kwds["style"] = wx.TAB_TRAVERSAL
        wx.Panel.__init__(self, *args, **kwds)
        self.info_btn = wx.Button(self, -1, "Get Info")
        self.kill_btn = wx.Button(self, -1, "Kill")
        self.process_list_ctrl = wx.ListCtrl(self, -1, style=wx.LC_REPORT|wx.SUNKEN_BORDER)

        self.__set_properties()
        self.__do_layout()

        self.Bind(wx.EVT_BUTTON, self.OnGetInfo, self.info_btn)
        # end wxGlade
        self.procManager = procManager
        self.Bind(wx.EVT_BUTTON, self.OnKill, self.kill_btn)

    def __set_properties(self):
        # begin wxGlade: ProcessManagerPanel.__set_properties
        pass
        # end wxGlade

    def __do_layout(self):
        # begin wxGlade: ProcessManagerPanel.__do_layout
        grid_sizer_1 = wx.FlexGridSizer(2, 1, 0, 0)
        grid_sizer_2 = wx.FlexGridSizer(1, 2, 0, 0)
        grid_sizer_2.Add(self.info_btn, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_2.Add(self.kill_btn, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_2.AddGrowableRow(0)
        grid_sizer_2.AddGrowableCol(0)
        grid_sizer_2.AddGrowableCol(1)
        grid_sizer_1.Add(grid_sizer_2, 1, wx.EXPAND, 0)
        grid_sizer_1.Add(self.process_list_ctrl, 1, wx.EXPAND, 0)
        self.SetSizer(grid_sizer_1)
        grid_sizer_1.Fit(self)
        grid_sizer_1.AddGrowableRow(1)
        grid_sizer_1.AddGrowableCol(0)
        # end wxGlade
        self.process_list_ctrl.InsertColumn(0,"PID", format = wx.LIST_FORMAT_CENTER, width = -1)
        self.process_list_ctrl.InsertColumn(1,"Calculation", format = wx.LIST_FORMAT_CENTER, width = -1)
        self.process_list_ctrl.InsertColumn(2,"Status", format = wx.LIST_FORMAT_CENTER, width = -1)
        size = self.process_list_ctrl.GetSize()
        size[0] = self.process_list_ctrl.GetColumnCount() * self.process_list_ctrl.GetColumnWidth(0) + 12
        self.GetParent().SetMinSize(size)

    def AddProcess(self, pid, typeStr, statusStr, infoCallBack = None):
        item = self.process_list_ctrl.InsertStringItem(0, str(pid))
        self.process_list_ctrl.SetStringItem(0,1, typeStr)
        self.process_list_ctrl.SetStringItem(0,2, statusStr)
    
    def removeProcess(self, pid):
        #Find the item number for the given pid
        self.process_list_ctrl.DeleteItem(self.process_list_ctrl.FindItem(-1, str(pid)))

    def OnGetInfo(self, event): # wxGlade: ProcessManagerPanel.<event_handler>
        """For "AnalyticDisp", "NumericDisp", or "AnalyticCrossSec" type processes, a window like that shown by
        the spinwave calc panel will be displayed which shows the version of the interaction and spin files that
        the process is using."""
        item = self.process_list_ctrl.GetFocusedItem()
        if item != -1: #If there is no focused item, -1 is returned.
            pid = int(self.process_list_ctrl.GetItemText(item))
            type, info = self.procManager.getProcessInfo(pid)
            if type=="AnalyticDisp" or type=="NumericDisp" or type=="AnalyticCrossSec":
                panel = showEditorWindow(self, "Files being used by process: " + str(pid), allowEditting = False)
                panel.loadInteractions(info[0])
                panel.loadSpins(info[1])
            if type=="Fit":
                showParamListFrame(info, str(pid) + " Fit Snapshot")
            else:
                print "Info for this type of process not implemented!"
            event.Skip()
        
    def OnKill(self, evt):
        item = self.process_list_ctrl.GetFocusedItem()
        if item != -1: #If there is no focused item, -1 is returned.
            pid = int(self.process_list_ctrl.GetItemText(item))
            self.procManager.killProcess(pid)
            self.removeProcess(pid)

# end of class ProcessManagerPanel


def ShowProcessesFrame(procManager):
    """Creates and displays a simple frame containing the ProcessManagerPanel."""
    
    frame = wx.Frame(procManager.parent, -1, title = "Processes")
    panel = ProcessManagerPanel(procManager, frame, -1)
    frame.Fit()
    frame.SetMinSize(frame.GetSize())
    frame.Show()
    return frame
        
#----Calculate Hsave-----------------------------------------------------------   
class CalcHsaveThread(Thread):
   def __init__ (self, parentWindow, queue, procManager):
      Thread.__init__(self)
      self.parent = parentWindow
      self.queue = queue
      self.procManager = procManager
    
   def run(self):
#       while(True):
           result = self.queue.get()
           pid = result[0]
           data = result[1]
           self.procManager.processDone(pid)
           self.procManager.startAnalyticDispersion(data[0])
           self.procManager.startNumericDispersion(*data)

def CalcHsaveFunc(queue, int_file, spin_file, direction, k_min, k_max, steps):
    pid = os.getpid()
    int_file = createFileCopy(int_file, pid, 0)
    spin_file = createFileCopy(spin_file, pid, 1)
    Hsave = spinwave_calc_file.driver1(spin_file, int_file)
    queue.put((pid, (Hsave, direction, k_min, k_max, steps)))
        
#----Analytic Dispersion-----------------------------------------------------------   
class AnalyticDispersionThread(Thread):
   def __init__ (self, parentWindow, queue, procManager):
      Thread.__init__(self)
      self.parent = parentWindow
      self.queue = queue
      self.procManager = procManager
    
   def run(self):
#       while(True):
           result = self.queue.get()
           pid = result[0]
           ans = result[1]
           self.procManager.processDone(pid)
           print 'giving wx a callback'
           wx.CallAfter(showAnalyticEigs, ans)
           #showAnalyticEigs(ans)


def AnalyticDispFunc(queue, Hsave):
    pid = os.getpid()
    myeigs=printing.eig_process(deepcopy(Hsave))
    queue.put((pid, printing.create_latex(myeigs, "eigs")))         
     
def showAnalyticEigs(ans):
    print 'creating latex frame'
    eig_frame = printing.LaTeXDisplayFrame(None, ans, 'Dispersion Eigenvalues')
    print 'showing latex frame'
    eig_frame.Show()
    print 'latex frame up'

#----Numeric Dispersion-------------------------------------------------------------   
class NumericDispersionThread(Thread):
    def __init__ (self, parentWindow, queue, procManager):
      Thread.__init__(self)
      self.parent = parentWindow
      self.queue = queue
      self.procManager = procManager
      
    def run(self):
#        while(flag):
            result = self.queue.get()
            pid = result[0]
            ans = result[1]
            self.procManager.processDone(pid)
            qrange = ans[0]
            wrange = ans[1]
            wx.CallAfter(showPlot, qrange, wrange)

def NumericDispFunc(queue, Hsave, direction, k_min, k_max, steps):
    pid = os.getpid()
    init_time = time.clock()
    qrange, wranges = spinwave_calc_file.driver2(Hsave, direction, steps, k_min, k_max)
    print "numeric cross section time; ", time.clock()-init_time, " seconds"
    queue.put((pid,(qrange, wranges)))

def showPlot(qrange, wrange):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for wrange1 in wrange:
        ax.plot(qrange, wrange1)
        plt.hold(True)
    plt.show()

#----Fitting---------------------------------------------------------------------
class FitThread(Thread):
    def __init__ (self, parentWindow, queue, procManager):
       Thread.__init__(self)
       self.parent = parentWindow
       self.queue = queue
       self.procManager = procManager
    
    def run(self):
        #poll the fit queue while there are still fitting processes running
        #while(len(self._fitProcesses)>0):#This could also be infinite
        while(True):
            result = self.queue.get()
            data = result[1]
            pid = result[0]
            self.procManager.processDone(pid)
            wx.CallAfter(showFitResultFrame,data, pid)

class FitSnapshotThread(Thread):
    def __init__(self, queue, procManager):
        Thread.__init__(self)
        self.procManager = procManager
        self.queue = queue
        
    def run(self):
        while(True):
            result = self.queue.get()
            data = result[1]
            pid = result[0]
            self.procManager.updateFit(pid, data)
            

class RecordKeeper():
    def __init__(self, pid, queue, queueFreq = 1, file = None, fileFreq =1):
        """Pid is the process ID of the fitting Process.  Queue is the queue
        to which the fitting process will add the parameter for th main process to read.
        The file is where the history of the parameter values will be recorded.  The two
        freq values are how often the parameter is processed, 1 being every time, 2 being
        every other time..."""
        if fileFreq < 1:
            fileFreq = 1
        if queueFreq < 1:
            queueFreq = 1
        
        self.pid = pid
        self.queue = queue
        self.queueFreq = queueFreq
        self.file = None
        if file:
            self.file = open(file, 'w')
        self.fileFreq = fileFreq
        self.qCount = 0
        self.fCount = 0
    
    def record(self, params):
        """This is a callback function called by the fitting routines 
        to track the progress of the fitters.  This function assumes
        the list of parameters, p is in order (p0,p1,p2) and assigns
        names accordingly."""
        #Create a list of tuples [(name, value), (name, value)...]
        list = []
        for i in range(len(params)):
            list.append(('p' + str(i), params[i]))
            
        if self.qCount < self.queueFreq:
            self.qCount += 1
        else:
            self.qCount = 0
            self.queue.put((self.pid,list))
            
        if self.file:
            if self.fCount < self.fileFreq:
                self.fCount += 1
            else:
                count = 0
                self.file.write("\n")
                for entry in list:
                    self.file.write(entry[0] + ": " + str(entry[1]) + "\n")
                self.file.flush()#In case the user wants to read while it is still being written to
           

def FitFunc(queue, sessionXML, fileName, k, tmin, tmax, size, tfactor, useMC, recordQueue):
    pid = os.getpid()
    sess = Session()
    sess.loadXMLStr(sessionXML)
    recordFile = tmpFileName(pid, 2)
    recordKeeper = RecordKeeper(pid, recordQueue, file = recordFile)
    ans = fitFromFile(fileName, sess, k = k, tMax = tmax, tMin = tmin, size = size, tFactor = tfactor, MCeveryTime = useMC, recordKeeper = recordKeeper.record)
    #The session which contains the fitted parameters is more useful
    queue.put((pid, sess.bondTable.data))
    
def AnnealFitFunc(queue, sessionXML, fileName, k, tmin, tmax, size, tfactor, useMC, recordQueue):
    pid = os.getpid()
    sess = Session()
    sess.loadXMLStr(sessionXML)
    recordFile = tmpFileName(pid, 2)
    recordKeeper = RecordKeeper(pid, recordQueue, file = recordFile)
    ans = annealFitFromFile(fileName, sess, k = k, tMax = tmax, tMin = tmin, size = size, tFactor = tfactor, MCeveryTime = useMC, recordKeeper = recordKeeper.record)
    #The session which contains the fitted parameters is more useful
    queue.put((pid, sess.bondTable.data))


    
#----Cross Section-------------------------------------------------
class CrossSectionThread(Thread):
    def __init__ (self, parent, queue, procManager):
        Thread.__init__(self)
        self.queue = queue
        self.parent = parent
        self.procManager = procManager

    def run(self):
#        while(True):
            result = self.queue.get()
            data = result[1]
            pid = result[0]
            self.procManager.processDone(pid)
            wx.CallAfter(load_plot, *data)

def CrossSectionFunc(queue, args):
    pid = os.getpid()
#    cs_driver(*args)
#    csection_arrays = np.load('csection_calc_data.npz')
#    csection = csection_arrays['csection'][0]
#    queue.put((pid,printing.create_latex(csection, "Cross section")))  #(PID, answer)
#    args = args[:-2]+(False, True)
#    if args == None:
#        raise Exception("Nonetype")
    outpath, colorbar_bool, plot_min, plot_max  = cs_driver(*args)
    queue.put((pid, (outpath, colorbar_bool, plot_min, plot_max)))

#def showCrossSectionFrame(parent, ans, pid):
#    #Almost the same as showAnalyticEigs()
#    eig_frame = printing.LaTeXDisplayFrame(parent, ans, 'Cross Section, PID: ' + str(pid))
#    eig_frame.Show()


        
#for testing
class App(wx.App):
    """Just to show the frame."""
    def __init__(self, redirect = False, filename = None):
        wx.App.__init__(self, redirect, filename)
    
    def OnInit(self):
        self.SetTopWindow(ShowProcessesFrame(ProcessManager()))
        return True


if __name__ == '__main__':       
    app = App(False)
    app.MainLoop()
    

