"""
Disclaimer
==========

This software was developed at the National Institute of Standards and Technology at the NIST Center for Neutron Research by employees of the Federal Government in the course of their official duties. Pursuant to title 17 section 105* of the United States Code this software is not subject to copyright protection and is in the public domain. The SPINAL software package is an experimental spinwave analysis system. NIST assumes no responsibility whatsoever for its use, and makes no guarantees, expressed or implied, about its quality, reliability, or any other characteristic. The use of certain trade names or commercial products does not imply any endorsement of a particular product, nor does it imply that the named product is necessarily the best product for the stated purpose. We would appreciate acknowledgment if the software is used.

*Subject matter of copyright: United States Government works

Copyright protection under this title is not available for any work of the United States Government, but the United States Government is not precluded from receiving and holding copyrights transferred to it by assignment, bequest, or otherwise."""


#!/usr/bin/env python

"""A Hello program in wxPython"""


import wx

class Frame(wx.Frame):
    """Frame class that displays image"""
    
    def __init__(self, image, parent = None, id = -1,
                 pos = wx.DefaultPosition,
                 title = 'Hello!'):
        """Create a frame instance and display an image"""
        temp = image.ConvertToBitmap()
        size = temp.GetWidth(), temp.GetHeight()
        wx.Frame.__init__(self, parent, id, title, pos, size)
        self.bmp = wx.StaticBitmap(parent = self, bitmap = temp)

class App(wx.App):
    """Application Class"""
    
    def OnInit(self):
        image = wx.Image('C:\\menu.jpg', wx.BITMAP_TYPE_JPEG)
        self.frame = Frame(image)
        sself.frame.Show()
        self.SetTopWindow(self.frame)
        return True

def main():
    app = App()
    app.MainLoop()
    
if __name__ == '__main__':
    main()