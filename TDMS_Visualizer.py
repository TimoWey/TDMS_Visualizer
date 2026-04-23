from nptdms import TdmsFile
from PyQt5.QtWidgets import QWidget, QGridLayout, QVBoxLayout, QHBoxLayout, QMainWindow, QApplication, QCheckBox, QComboBox, QPushButton, QColorDialog, QLabel, QScrollArea, QFileDialog, QInputDialog, QLineEdit, QSizePolicy
from PyQt5.QtCore import Qt, QObject, QEvent, QTimer, QSize
from PyQt5.QtGui import QFont
import html
import numpy as np
import pyqtgraph as pg

# With OpenGL, the plot viewport can composite incorrectly and paint over the
# sidebar on some Windows/HiDPI setups; keep 2D rendering for reliable clipping.
pg.setConfigOptions(useOpenGL=False)
import sys
from functools import partial
import re

# Minimum horizontal space reserved for the plot; must match _CappedMinPlotWidget cap + MainWindow.
PLOT_MIN_W = 200


def _axis_lbl_to_html(s):
    """
    pyqtgraph AxisItem injects label text into a <span> with setHtml; characters like
    '&' and '<' in TDMS / channel names would break the markup or render as garbled
    lines unless escaped.
    """
    if s is None:
        return ""
    return html.escape(str(s), quote=False)

class TDMS_File():
    def __init__(self, file_open):
        self.tdms_file = TdmsFile.read(file_open)
        self.start_time_info = (None, 15, "")
        self.time_range = 0
        self.timesDict = self.setTimes()
        self.nondigital = list()
        self.digital = list()
        self.digital = self.findDigital()
        self.editDigital()
        self.file_name = file_open
        # self.debug()

    def _group_has_time_channel(self, group):
        try:
            group["time"]
            return True
        except Exception:
            return False

    def _channel_time_axis_seconds(self, channel, global_start_datetime64):
        """
        Returns seconds-from-global-start for this channel.

        Supports TDMS waveform channels that provide:
        - wf_start_time: np.datetime64
        - wf_start_offset: seconds
        - wf_increment: seconds/sample
        """
        n = len(channel)
        if n <= 0:
            return np.array([], dtype=float)

        props = getattr(channel, "properties", {}) or {}
        wf_inc = props.get("wf_increment", None)
        wf_start_offset = props.get("wf_start_offset", 0.0)
        wf_start_time = props.get("wf_start_time", None)

        if wf_start_time is None or wf_inc is None or global_start_datetime64 is None:
            # Fallback: sample index as "time"
            return np.arange(n, dtype=float)

        try:
            start_delta_s = (wf_start_time - global_start_datetime64) / np.timedelta64(1, "s")
            start_s = float(start_delta_s) + float(wf_start_offset)
            return start_s + np.arange(n, dtype=float) * float(wf_inc)
        except Exception:
            return np.arange(n, dtype=float)
    
    def setTimes(self):
        # Two supported TDMS layouts:
        # 1) Legacy: each group contains a channel literally named "time"
        # 2) Waveform: channels carry timing via wf_* properties (wf_start_time, wf_increment, ...)

        timesDict = {}
        groups = self.tdms_file.groups()
        if not groups:
            self.time_range = 0
            return timesDict

        if all(self._group_has_time_channel(g) for g in groups):
            # Existing behavior
            self.start_time_info = self.find_earliest_time()
            if self.start_time_info[0] is True:
                self.time_range = self.start_time_info[3] - self.start_time_info[1]
                start_time = self.start_time_info[1]
                # time is formatted as a float
                for group in groups:
                    time = group["time"][:]
                    for i, times in enumerate(time):
                        time[i] = float(times) - start_time
                    timesDict[group.name] = time
            else:  # string
                start = self.start_time_info[1]
                start_time = int(start[6:8]) + int(start[10:12]) / 100  # seconds, milliseconds
                end = self.start_time_info[3]
                end_time = int(end[6:8]) + int(end[10:12]) / 100
                self.time_range = end_time - start_time
                for group in groups:
                    time = group["time"][:]
                    for i, times in enumerate(time):
                        seconds = times[6:8]
                        milliseconds = times[10:12]
                        new_time = int(seconds) + int(milliseconds) / 100
                        time[i] = new_time - start_time
                    timesDict[group.name] = time
            return timesDict

        # Waveform layout
        global_start = None
        global_end_s = 0.0
        for group in groups:
            for channel in group.channels():
                props = getattr(channel, "properties", {}) or {}
                wf_start_time = props.get("wf_start_time", None)
                if wf_start_time is None:
                    continue
                if global_start is None or wf_start_time < global_start:
                    global_start = wf_start_time

        # If we can't find any wf_start_time, we still allow plotting using sample index
        for group in groups:
            for channel in group.channels():
                t = self._channel_time_axis_seconds(channel, global_start)
                timesDict[(group.name, channel.name)] = t
                if len(t) > 0:
                    global_end_s = max(global_end_s, float(t[-1]))

        self.start_time_info = (True, 0.0, "waveform", global_end_s)
        self.time_range = global_end_s
        return timesDict

    def find_earliest_time(self): # it's really finding both earliest and latest times
        earliest_time = self.tdms_file.groups()[0]["time"][0]
        earliest_group = self.tdms_file.groups()[0].name
        for group in self.tdms_file.groups():
            first_time = group["time"][0]
            if first_time < earliest_time:
                earliest_time = first_time
                earliest_group = group.name
        latest_time_list = [group["time"][-1] for group in self.tdms_file.groups()]
        latest_time = max(latest_time_list)
        # print(latest_time)
        # print(earliest_time)
        try:
            earliest_time = float(earliest_time)
            return (True, earliest_time, earliest_group, float(latest_time))
        except:
            return (False, earliest_time, earliest_group, latest_time)
    
    def findDigital(self):
        digital = []
        breakOut = False
        for group in self.tdms_file.groups():
            for channel in group.channels():
                if channel.name == "time": continue
                for data in channel[:]:
                    data = float(data)
                    if abs(data) >= 1.0e-6 and abs(data - 1) >= 1.0e-6:
                        breakOut = True
                        break
            if breakOut:
                breakOut = False
                self.nondigital.append(group.name)
            else:
                digital.append(group.name)
        return digital
    
    def editDigital(self): # make the digital plots more asymptomic
        for digital_group in self.digital:
            for channel in self.tdms_file[digital_group].channels():
                if channel.name == "time":
                    continue

                # time axis may be stored per-group (legacy) or per-channel (waveform)
                if digital_group in self.timesDict:
                    time_info = self.timesDict[digital_group]
                else:
                    time_info = self.timesDict.get((digital_group, channel.name), None)
                    if time_info is None:
                        continue

                for i in range(1, len(channel[1:])):
                    if abs(float(channel[i]) - float(channel[i - 1])) > 1.0e-6:  # valve opened or closed
                        time_info[i - 1] = time_info[i] - 0.001
        
        '''digital_channel_count = 0
        for digital_group in self.digital:
            digital_channel_count += len(self.tdms_file[digital_group].channels())
        digital_channel_count -= len(self.digital) # subtracts one per group (to ignore time channel, which really isn't a channel)
        print("Number of Digital Channels:", digital_channel_count)'''

        self.digital_adjustment = {}
        adjustment = 0
        for digital_group in self.digital:
            for digital_channel in self.tdms_file[digital_group].channels():
                if digital_channel.name == "time": continue
                self.digital_adjustment[digital_channel.name] = adjustment
                adjustment += 0.01
        #print(self.digital_adjustment)
    
    def debug(self):
        print("---------Verify with Excel Spreadsheet---------")
        print("Group Names:", [group.name for group in self.tdms_file.groups()])
        print("Digital Groups: (Only 0 and 1 for output)", self.digital)
        print()
        print("** Start time Info **")
        print("Start time:", self.start_time_info[1])
        print("Time was from", self.start_time_info[2], "group")


class _CappedMinPlotWidget(pg.PlotWidget):
    """
    PyQtGraph's default minimumSizeHint is often much wider than the space next to
    a fixed-width sidebar; when the user shrinks the window, Qt can still allocate
    the full hinted width to the plot and the panels overlap. Cap the *hint* width
    so the layout can shrink the plot; enforce a hard floor via setMinimumWidth.
    """

    def minimumSizeHint(self):
        s = super().minimumSizeHint()
        w = s.width() if s.width() > 0 else PLOT_MIN_W
        w = min(w, PLOT_MIN_W)
        h = s.height() if s.height() > 0 else 100
        return QSize(w, max(h, 100))


class _PlotViewportAxisDblClickFilter(QObject):
    """
    QGraphicsView does not always deliver dbl-clicks to AxisItem filters reliably.
    We watch the plot viewport, map the click to the scene, and test axis bounding rects.
    """

    def __init__(self, main_window):
        super().__init__(main_window)
        self._mw = main_window

    def eventFilter(self, obj, event):
        if event.type() != QEvent.MouseButtonDblClick or event.button() != Qt.LeftButton:
            return False
        if obj is not self._mw.graphWidget.viewport():
            return False
        pt = self._mw.graphWidget.mapToScene(event.pos())
        tl = self._mw.graphWidget.plotItem.titleLabel
        if tl is not None and tl.isVisible() and tl.scene() is not None:
            if tl.sceneBoundingRect().contains(pt):
                self._mw._prompt_edit_plot_title()
                return True
        for axis in self._mw._editable_axes:
            if axis is None or axis.scene() is None:
                continue
            if axis.sceneBoundingRect().contains(pt):
                self._mw._prompt_edit_axis_label(axis)
                return True
        return False


class MainWindow(QMainWindow):

    def __init__(self, tdms):
        super(MainWindow, self).__init__()
        self.tdms = tdms
        # TDMS_Viewer-like normalization/bundling:
        # - groups like "Loop [3]" -> "Loop"
        # - channels like "Torque 12:34:56.789" -> "Torque"
        self._TRAILING_TIME_RE = re.compile(r"\s+\d{2}:\d{2}:\d{2}\.\d{3}$")
        self._TRAILING_INDEX_RE = re.compile(r"\s*\[\d+\]$")

        self._raw_groups_by_norm = {}
        self._combined_by_norm = {}  # (normGroup, normChannel) -> list[(rawGroup, rawChannel)]
        self._norm_channels_by_norm_group = {}  # normGroup -> sorted list[normChannel]
        self._build_normalized_index()

        self.legendFontSize = 7

        self.setWindowTitle("TREL TDMS Data Visualization: " + tdms.file_name)

        self._plot_min_w = PLOT_MIN_W
        # Wider panel = less width left for the plot (smaller plot area on the same screen).
        self._sidebar_w = 560

        self.graphWidget = _CappedMinPlotWidget()
        self.graphWidget.setMinimumWidth(self._plot_min_w)
        # Ensure the scene is clipped to the viewport (sibling widgets stack correctly).
        self.graphWidget.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.graphWidget.viewport().setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.initGraphWidget()

        self.graphWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        widthOfRight = self._sidebar_w

        sidebar = QWidget()
        sidebar.setFixedWidth(widthOfRight)
        sidebar.setMinimumWidth(widthOfRight)
        sidebar.setMaximumWidth(widthOfRight)
        sidebar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        sidebar.setAutoFillBackground(True)
        sidebar_l = QVBoxLayout(sidebar)
        sidebar_l.setContentsMargins(0, 0, 0, 0)
        sidebar_l.setSpacing(0)

        label = QLabel("Groups within tdms file: \n (contained in drop-down menu)")
        label.setAlignment(Qt.AlignCenter)
        label.setFixedSize(widthOfRight, 45)
        label.setStyleSheet("border: 1px solid black")
        sidebar_l.addWidget(label)

        self.comboBox = QComboBox(self)
        self.comboBox.setMaximumWidth(widthOfRight)
        sidebar_l.addWidget(self.comboBox)

        spacer = QLabel()
        spacer.setFixedSize(widthOfRight, 10)
        sidebar_l.addWidget(spacer)

        label2 = QLabel("Channels within selected group: \n Use checkboxes to plot/unplot \n Legend is in top left corner")
        label2.setAlignment(Qt.AlignCenter)
        label2.setFixedSize(widthOfRight, 55)
        label2.setStyleSheet("border: 1px solid black")
        sidebar_l.addWidget(label2)

        self.buttonLayout = QVBoxLayout()
        self.buttonLayout.setAlignment(Qt.AlignTop)

        scroll = QScrollArea()
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setWidgetResizable(True)
        scroll.setMaximumSize(widthOfRight, 250)
        scroll.setMaximumWidth(widthOfRight)

        self.buttonWidget = QWidget()
        self.buttonWidget.setLayout(self.buttonLayout)
        scroll.setWidget(self.buttonWidget)
        sidebar_l.addWidget(scroll)

        spacer2 = QLabel()
        spacer2.setFixedSize(widthOfRight, 10)
        sidebar_l.addWidget(spacer2)

        label3 = QLabel("**Interactions** \n Col 1: Plotted Channels \n Col 2: Color \n Col 3: Point style \n Col 4: Secondary Y-axis (non-digital only)")
        label3.setAlignment(Qt.AlignCenter)
        label3.setFixedSize(widthOfRight, 100)
        label3.setStyleSheet("border: 1px solid black")
        sidebar_l.addWidget(label3)

        scroll2 = QScrollArea()
        scroll2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll2.setWidgetResizable(True)
        scroll2.setMaximumSize(widthOfRight, 250)
        scroll2.setMaximumWidth(widthOfRight)

        self.interLayout = QGridLayout()
        self.interLayout.setAlignment(Qt.AlignTop)
        self.interaction = QWidget()
        self.interaction.setLayout(self.interLayout)
        scroll2.setWidget(self.interaction)
        sidebar_l.addWidget(scroll2)

        # Horizontal box layout strictly divides width: plot never shares pixels with the panel.
        central = QWidget()
        central_h = QHBoxLayout(central)
        central_h.setContentsMargins(0, 0, 0, 0)
        central_h.setSpacing(0)
        central_h.addWidget(self.graphWidget, 1)
        central_h.addWidget(sidebar, 0)

        # No overlap: the window must stay at least as wide as sidebar + minimum plot
        # (see _CappedMinPlotWidget: pyqtgraph's own hint is ignored for the sum).
        central.setMinimumWidth(self._sidebar_w + self._plot_min_w)

        self.plottedData = {}
        self.initComboBox()
        # Default to the first normalized group
        first_norm = next(iter(self._raw_groups_by_norm.keys()), None)
        if first_norm is not None:
            self.setButtonLayout(first_norm)

        self.setCentralWidget(central)
        self.showMaximized()

    def _normalize_group_name(self, name: str) -> str:
        return self._TRAILING_INDEX_RE.sub("", str(name)).strip()

    def _normalize_channel_name(self, name: str) -> str:
        return self._TRAILING_TIME_RE.sub("", str(name)).strip()

    def _build_normalized_index(self):
        raw_groups_by_norm = {}
        combined = {}
        norm_channels_by_group = {}

        for g in self.tdms.tdms_file.groups():
            ng = self._normalize_group_name(g.name)
            raw_groups_by_norm.setdefault(ng, []).append(g.name)
            for ch in g.channels():
                if ch.name == "time":
                    continue
                nc = self._normalize_channel_name(ch.name)
                combined.setdefault((ng, nc), []).append((g.name, ch.name))

        for ng in raw_groups_by_norm.keys():
            chans = sorted({nc for (gk, nc) in combined.keys() if gk == ng}, key=str.lower)
            norm_channels_by_group[ng] = chans

        self._raw_groups_by_norm = dict(sorted(raw_groups_by_norm.items(), key=lambda kv: str(kv[0]).lower()))
        self._combined_by_norm = combined
        self._norm_channels_by_norm_group = norm_channels_by_group

    def initComboBox(self):
        for norm_group in self._raw_groups_by_norm.keys():
            self.comboBox.addItem(norm_group)
        self.comboBox.setMinimumContentsLength(25)
        self.comboBox.currentIndexChanged.connect(self.comboBoxChange)

    def comboBoxChange(self):
        # print("Current selection:", self.comboBox.currentText())
        self.changeButtons(self.comboBox.currentText())

    def changeButtons(self, groupText):
        for i in reversed(range(self.buttonLayout.count())): # deletes all checkBoxes
            self.buttonLayout.itemAt(i).widget().deleteLater()
        self.setButtonLayout(groupText)
    
    def setButtonLayout(self, groupText):
        # groupText is a *normalized group* shown in the dropdown
        for norm_ch in self._norm_channels_by_norm_group.get(groupText, []):
            cb = QCheckBox(norm_ch)
            if (groupText, norm_ch) in self.plottedData:
                cb.setChecked(True)
            self.buttonLayout.addWidget(cb)
            cb.stateChanged.connect(partial(self.clicked, groupText, norm_ch))

    def clicked(self, groupText, channel):
        # groupText is a normalized group, channel is a normalized channel.
        key = (groupText, channel)
        if key in self.plottedData:  # deselect
            plotInfo = self.plottedData.pop(key)
            item = plotInfo[0]
            if groupText in self.tdms.digital:
                self.p2.removeItem(item)
                try:
                    self.legend.removeItem(item)
                except Exception:
                    pass
            else:
                if len(plotInfo) > 4 and plotInfo[4].isChecked():
                    self.p2.removeItem(item)
                else:
                    self.graphWidget.removeItem(item)
            plotInfo[1].deleteLater()
            plotInfo[2].deleteLater()
            plotInfo[3].deleteLater()
            if len(plotInfo) > 4:
                plotInfo[4].deleteLater()
            return

        # select: combine all matching raw channels into a single curve (TDMS_Viewer style)
        entries = self._combined_by_norm.get(key, [])
        segments = []  # list[(x, y)]
        any_digital = False
        any_nondigital = False
        for raw_g, raw_ch in entries:
            if raw_g in self.tdms.digital:
                any_digital = True
            else:
                any_nondigital = True
            try:
                if raw_g in self.tdms.timesDict:
                    x = self.convert_to_np(self.tdms.timesDict[raw_g])
                else:
                    x = self.convert_to_np(self.tdms.timesDict[(raw_g, raw_ch)])
                y = self.convert_to_np(self.tdms.tdms_file[raw_g][raw_ch][:])
                if raw_g in self.tdms.digital:
                    y = self.adjust_digital(raw_ch, y)
            except Exception:
                continue
            if x.size and y.size:
                x = x.astype(float, copy=False)
                segments.append((x, y))

        if not segments:
            return

        # Order segments by their own start time, then concatenate with NaN separators
        # to prevent pyqtgraph from drawing long lines between distant segments.
        def _seg_start(seg):
            x, _ = seg
            if x.size == 0:
                return float("inf")
            finite = x[np.isfinite(x)]
            return float(finite[0]) if finite.size else float("inf")

        segments.sort(key=_seg_start)

        xs = []
        ys = []
        for i, (x, y) in enumerate(segments):
            if i > 0:
                xs.append(np.array([np.nan], dtype=float))
                ys.append(np.array([np.nan], dtype=float))
            xs.append(x)
            ys.append(y)

        x_plot = np.concatenate(xs)
        y_plot = np.concatenate(ys)

        legend_name = f"{groupText}: {channel}"
        # If mixed digital/non-digital, keep on left axis (non-digital) to avoid forcing right.
        plot_on_right = (any_digital and not any_nondigital)
        if plot_on_right:
            item = pg.PlotDataItem(
                x=x_plot,
                y=y_plot,
                symbolBrush=pg.mkBrush('r'),
                symbol=None,
                pen=pg.mkPen('r', width=1),
                name=legend_name,
            )
            self.p2.addItem(item)
            self.legend.addItem(item, legend_name)
        else:
            item = self.graphWidget.plot(
                x=x_plot,
                y=y_plot,
                symbolBrush=pg.mkBrush('r'),
                symbol=None,
                pen=pg.mkPen('r', width=1),
                name=legend_name,
            )

        self.plottedData[key] = item
        self.j += 1
        self.setInteraction(groupText, channel, item, self.j)

    def initGraphWidget(self):
        self.j = -1
        self.graphWidget.setBackground('w')
        pg.setConfigOption('foreground', 'k')
        self.graphWidget.setTitle("Data Visualization of " + self.tdms.file_name, color="k", size="16px")
        self.legend = self.graphWidget.addLegend(offset=(100, 40), labelTextSize=str(self.legendFontSize) + "pt")
        self.graphWidget.showGrid(x=True, y=True)
        styles = {"color": "#222222", "font-size": "14px"}
        self._axis_label_style = dict(styles)
        self.graphWidget.setLabel("bottom", _axis_lbl_to_html("Time"), **styles)
        if self.tdms.nondigital:
            glist = ", ".join(self.tdms.nondigital)
            leftPlain = f"Non-digital ({glist})"
        else:
            leftPlain = "Y (left)"
        self.graphWidget.setLabel("left", _axis_lbl_to_html(leftPlain), **styles)
        self.graphWidget.plotItem.vb.setRange(xRange = (0, self.tdms.time_range))
        self.graphWidget.enableAutoRange(axis='y')
        self.graphWidget.setAutoVisible(y=True)

        p1 = self.graphWidget.plotItem
        # Single secondary (right) y-axis used for:
        #   - digital channels (0/1) which are always plotted on the right
        #   - non-digital channels the user opts-in via the right-axis checkbox
        self.p2 = pg.ViewBox()
        self.p2.setZValue(-0.5)
        self.graphWidget.showAxis('right')
        p1.scene().addItem(self.p2)
        self.graphWidget.getAxis('right').linkToView(self.p2)
        self.p2.setXLink(self.graphWidget)
        self.graphWidget.getAxis("right").setLabel(
            _axis_lbl_to_html("Secondary Y-axis"), **styles
        )
        self.p2.enableAutoRange(axis=pg.ViewBox.YAxis)
        self.p2.setYRange(0, 1, padding=0.05)

        self._editable_axes = [
            self.graphWidget.getAxis("bottom"),
            self.graphWidget.getAxis("left"),
            self.graphWidget.getAxis("right"),
        ]
        self._plot_viewport_dblclick_filter = _PlotViewportAxisDblClickFilter(self)
        self.graphWidget.viewport().installEventFilter(self._plot_viewport_dblclick_filter)

        def _updateAuxViews():
            rect = p1.vb.sceneBoundingRect()
            self.p2.setGeometry(rect)
            self.p2.linkedViewChanged(p1.vb, self.p2.XAxis)

        _updateAuxViews()
        p1.vb.sigResized.connect(_updateAuxViews)

        self.vLine = pg.InfiniteLine(pos = self.tdms.time_range, movable=False, labelOpts = {'rotateAxis':(-1,0)}, 
                                    label = "Last Data Record \nat " + str(format(self.tdms.time_range, ",.2f")) + " seconds")
        font = QFont()
        font.setPixelSize(11)
        self.vLine.label.setFont(font)
        self.vLine.label.setPosition(0.84)
        self.graphWidget.addItem(self.vLine)

        self.graphWidget.plotItem.vb.sigXRangeChanged.connect(self.setYRange)

    def setYRange(self):
        self.graphWidget.enableAutoRange(axis='y')
        self.graphWidget.setAutoVisible(y=True)
        self.p2.enableAutoRange(axis=pg.ViewBox.YAxis)

    def _prompt_edit_axis_label(self, axis):
        raw = (axis.labelText or "").strip()
        current = html.unescape(raw)
        text, ok = QInputDialog.getText(
            self, "Edit axis label", "Label text:", QLineEdit.Normal, current
        )
        if not ok:
            return
        k = self._axis_label_style
        axis.setLabel(
            _axis_lbl_to_html((text or "").strip()),
            units=axis.labelUnits,
            unitPrefix=axis.labelUnitPrefix,
            unitPower=axis.unitPower,
            siPrefixEnableRanges=axis.getSIPrefixEnableRanges(),
            **k,
        )

    def _prompt_edit_plot_title(self):
        raw = (self.graphWidget.plotItem.titleLabel.text or "").strip()
        current = html.unescape(raw)
        text, ok = QInputDialog.getText(
            self, "Edit plot title", "Title text:", QLineEdit.Normal, current
        )
        if not ok:
            return
        t = (text or "").strip()
        if not t:
            self.graphWidget.setTitle(None)
        else:
            self.graphWidget.setTitle(
                _axis_lbl_to_html(t), color="k", size="16px"
            )

    def add_to_plot(self, groupText, channelText, display_name=None, add_legend=True):
        if groupText in self.tdms.timesDict:
            time = self.tdms.timesDict[groupText]
        else:
            time = self.tdms.timesDict[(groupText, channelText)]
        channel_info = self.tdms.tdms_file[groupText][channelText][:]
        np_channel_info = self.convert_to_np(channel_info)
        # print(groupText)
        name = display_name if display_name is not None else (groupText + ": " + channelText)
        if groupText in self.tdms.digital:
            np_channel_info = self.adjust_digital(channelText, np_channel_info)
            plotDataItem = pg.PlotDataItem(x = self.convert_to_np(time), y = np_channel_info, symbolBrush = pg.mkBrush('r'),
                                             symbol = None, pen=pg.mkPen('r', width = 1), name = name)
            self.p2.addItem(plotDataItem)
            if add_legend and name is not None:
                self.legend.addItem(plotDataItem, name)
        else:
            plotDataItem = self.graphWidget.plot(x = self.convert_to_np(time), y = np_channel_info, symbolBrush = pg.mkBrush('r'),
                                             symbol = None, pen=pg.mkPen('r', width = 1), name = name if add_legend else None)
        return plotDataItem

    def convert_to_np(self, data):
        return np.array(data, dtype = float)
    
    def adjust_digital(self, channel, np_channel_info):
        for i in range(len(np_channel_info)):
            if abs(float(np_channel_info[i]) - 1) < 1.0e-6: # 1 
                np_channel_info[i] += self.tdms.digital_adjustment[channel]
            else: # 0
                np_channel_info[i] -= self.tdms.digital_adjustment[channel]
        return np_channel_info

    def setInteraction(self, group, channel, plotDataItem, i):
        label = QLabel(group + ":  " + channel) # normalized group - normalized channel
        self.interLayout.addWidget(label, i, 0)

        button = QPushButton('Color', self)
        button.clicked.connect(partial(self.set_color, group, channel))
        self.interLayout.addWidget(button, i, 1)

        comboBox = QComboBox()
        [comboBox.addItems(['No Points', 'Circles', 'Squares', 'Triangles', 'Diamonds', 'Pluses'])]
        self.shapes = { 0 : None, 1 : 'o', 2 : 's', 3 : 't', 4 : 'd', 5: '+'}
        comboBox.setCurrentIndex(0)
        comboBox.currentIndexChanged.connect(partial(self.set_shape, group, channel))
        self.interLayout.addWidget(comboBox, i, 2)

        axis_cb = QCheckBox("")
        axis_cb.blockSignals(True)
        # normalized group may map to raw digital groups; disable only if all raw groups are digital
        raw_groups = self._raw_groups_by_norm.get(group, [])
        all_digital = bool(raw_groups) and all(rg in self.tdms.digital for rg in raw_groups)
        if all_digital:
            axis_cb.setEnabled(False)
            axis_cb.setChecked(True)
        else:
            axis_cb.setChecked(False)
        axis_cb.blockSignals(False)
        axis_cb.stateChanged.connect(partial(self.toggle_y_axis, group, channel))
        self.interLayout.addWidget(axis_cb, i, 3)

        self.plottedData[(group, channel)] = (plotDataItem, label, button, comboBox, axis_cb)

    def toggle_y_axis(self, group, channel, state):
        if (group, channel) not in self.plottedData:
            return
        # Disable only if all raw groups in this normalized group are digital.
        raw_groups = self._raw_groups_by_norm.get(group, [])
        all_digital = bool(raw_groups) and all(rg in self.tdms.digital for rg in raw_groups)
        if all_digital:
            return
        main_vb = self.graphWidget.getPlotItem().getViewBox()
        use_right = (state == Qt.Checked)
        item = self.plottedData[(group, channel)][0]
        if use_right:
            main_vb.removeItem(item)
            self.p2.addItem(item)
        else:
            self.p2.removeItem(item)
            main_vb.addItem(item)
        self.p2.enableAutoRange(axis=pg.ViewBox.YAxis)
        self.p2.updateAutoRange()
        self.graphWidget.enableAutoRange(axis='y')

    def set_color(self, group, channel):
        color = QColorDialog.getColor()
        item = self.plottedData[(group, channel)][0]
        item.setSymbolBrush(color)
        item.setPen(color, width = 1)
    
    def set_shape(self, group, channel):
        shape_index = self.plottedData[(group, channel)][3].currentIndex()
        item = self.plottedData[(group, channel)][0]
        item.setSymbol(self.shapes[shape_index])

def main():
    app = QApplication(sys.argv)
    file_to_open, _ = QFileDialog.getOpenFileName(
        None,
        "Select TDMS file",
        "",
        "TDMS files (*.tdms *.dat *.dat.*.tdms);;All files (*.*)",
    )
    if not file_to_open:
        return

    tdms = TDMS_File(file_to_open)
    main_window = MainWindow(tdms)
    sys.exit(app.exec_())

def run_app(tdms):
    app = QApplication(sys.argv)
    main = MainWindow(tdms)
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
