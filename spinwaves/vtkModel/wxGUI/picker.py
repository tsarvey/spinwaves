"""
Disclaimer
==========

This software was developed at the National Institute of Standards and Technology at the NIST Center for Neutron Research by employees of the Federal Government in the course of their official duties. Pursuant to title 17 section 105* of the United States Code this software is not subject to copyright protection and is in the public domain. The SPINAL software package is an experimental spinwave analysis system. NIST assumes no responsibility whatsoever for its use, and makes no guarantees, expressed or implied, about its quality, reliability, or any other characteristic. The use of certain trade names or commercial products does not imply any endorsement of a particular product, nor does it imply that the named product is necessarily the best product for the stated purpose. We would appreciate acknowledgment if the software is used.

*Subject matter of copyright: United States Government works

Copyright protection under this title is not available for any work of the United States Government, but the United States Government is not precluded from receiving and holding copyrights transferred to it by assignment, bequest, or otherwise."""


import vtk
from wx.py.dispatcher import send
from Atom_Preferences import openAtomDisplayPref
from spinwaves.vtkModel.AtomClass import Atom
from wx.py.dispatcher import connect, send

class Picker():
    
    """This class is in charge of listening to pick events in the render window
    and handling them.  The actor is highlighted by turning the ambient lighting
    on.  Then the object (Atom or Bond; others are not supported) is found using
    the vtkDrawer.  The object's __str__ is printed"""

    def __init__(self, vtkDrawer, iren, renderer):
        """ vtkDrawer is an instance of VTKDrawer
        iren is a render window interactor."""
        self.highlight_level = .8 #The ambient lighting on selected items
        
        self.iren = iren
        self.ren1 = renderer
        self.drawer = vtkDrawer
        
        #set the picker so props can be picked
        self.picker = vtk.vtkPropPicker()
        iren.SetPicker(self.picker)
        
        #Add my own pick function
        self.observerNum = iren.AddObserver("LeftButtonPressEvent", self.pick)
        
        self.SelectedActor = None   #used by the pick() so that only one item is
                                    #picked at a time and it is not repicked
                                    
        #Right clicking will popup a window with display settings
        self.right_observerNum = iren.AddObserver("RightButtonPressEvent", self.rightClick)
    
    def pick(self, obj, event):
        Mouse_Position = self.iren.GetEventPosition()
        self.picker.PickProp(Mouse_Position[0],Mouse_Position[1], self.ren1)
        if(self.SelectedActor == self.picker.GetActor()): #the actor is already picked
            return
        if(self.SelectedActor != None):
            self.SelectedActor.GetProperty().SetAmbient(0) #unhighlight old picked actor
        self.SelectedActor = self.picker.GetActor()
        if self.SelectedActor != None:
            self.SelectedActor.GetProperty().SetAmbient(self.highlight_level)#make the selected actor stand out
            self.iren.GetRenderWindow().Render()
            #find the Atom at this position and print its description
            modelObj = self.drawer.getObjFromActor(self.SelectedActor)
#            print "sending signal pick..."
            send(signal = "Pick Event", sender = "Picker", obj = modelObj)
            actors = self.ren1.GetActors()
            
            #print modelObj
            #Rather than printing this info to stdout, I will send it as a signal and the GUI
            #can intercept it and display it where it wants, such as the status bar.
            #send("Actor Selected", sender = "Picker", description = modelObj.__str__())
            #There is already a signal sent, "Pick Event"
            

    
    
    def rightClick(self, obj, event):
        Mouse_Position = self.iren.GetEventPosition()
        self.picker.PickProp(Mouse_Position[0],Mouse_Position[1], self.ren1)
        obj = self.drawer.getObjFromActor(self.picker.GetActor())
        if obj.__class__.__name__ == Atom.__name__:
            #The actor that was clicked on represents an atom (obj is an instance of Atom)
            openAtomDisplayPref(obj.getElementSymbol())

    def removeObserver(self):
        """removes this picker from the render window
        If this is not called and another picker is added, both will be active"""
        
        self.iren.RemoveObserver(self.observerNum)
        self.iren.RemoveObserver(self.right_observerNum)
    
    def getPicked(self):
        return self.SelectedActor
