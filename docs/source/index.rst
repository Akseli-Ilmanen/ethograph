:hide-toc:

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started/index
   advanced/index
   examples/index
   api_index
   community/index

ethograph
=========

EthoGraph is a graphical user interface for visualizing and segmenting
multimodal timeseries behavioural data. It builds upon a number of :ref:`open-source
libraries <target-support>` to load and quickly render video and pose files, audio and
spectrograms, ephys recordings in various formats, and arbitrary
multi-dimensional timeseries — all on one shared timeline.

Note this GUI is still in development. I welcome people testing it and
:doc:`providing feedback <community/contributing>`!

.. raw:: html

   <style>
     .vimeo-wrapper { width: 100%; }
     @media (min-width: 768px) {
       .vimeo-wrapper { width: 65%; margin: 0 auto; }
     }
   </style>
   <div class="vimeo-wrapper">
     <div style="padding:56.25% 0 0 0;position:relative;"><iframe src="https://player.vimeo.com/video/1206424641?badge=0&amp;autopause=0&amp;player_id=0&amp;app_id=58479" frameborder="0" allow="autoplay; fullscreen; picture-in-picture; clipboard-write; encrypted-media; web-share" referrerpolicy="strict-origin-when-cross-origin" style="position:absolute;top:0;left:0;width:100%;height:100%;" title="Ethograph Demo"></iframe></div>
   </div>
   <script src="https://player.vimeo.com/api/player.js"></script>

Quickstart
----------

.. raw:: html

   <style>
   .dcbtn {
     display: inline-flex; align-items: center; gap: .5rem;
     padding: .45rem .9rem; margin: .5rem 0 1rem;
     border: 1px solid var(--pst-color-border, #e0e0e0); border-radius: 6px;
     background: var(--pst-color-surface, #f5f5f5);
     color: var(--pst-color-text-base, #212121);
     font-size: .88rem; font-weight: 600; font-family: inherit; cursor: pointer;
     transition: border-color .15s, background .15s;
   }
   .dcbtn:hover { border-color: var(--pst-color-primary, #00897b); background: rgba(0,137,123,.08); }

   .dcov {
     display: none; position: fixed; inset: 0; z-index: 1200;
     background: rgba(0,0,0,.45); padding: 2rem 1rem; overflow-y: auto;
   }
   .dcov.won { display: block; }
   .dcmod {
     max-width: 560px; margin: 3rem auto;
     background: var(--pst-color-background, #fff);
     border: 1px solid var(--pst-color-border, #e0e0e0);
     border-radius: 10px; padding: 1.5rem 1.6rem;
     box-shadow: 0 8px 32px rgba(0,0,0,.25);
   }
   .dchd { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.2rem; }
   .dchd h3 { margin: 0; font-size: 1.05rem; }
   .dcx { background: none; border: none; font-size: 1.4rem; line-height: 1; cursor: pointer;
          color: var(--pst-color-text-muted, #757575); padding: 0 .2rem; font-family: inherit; }

   .dcpg { display: none; }
   .dcpg.won { display: block; }
   .dcqt { font-size: 1rem; font-weight: 600; margin-bottom: .35rem; }
   .dcqh { font-size: .85rem; color: var(--pst-color-text-muted, #757575); margin-bottom: 1rem; line-height: 1.5; }

   .dcch { display: flex; flex-direction: column; gap: .45rem; }
   .dcc {
     display: flex; align-items: flex-start; gap: .7rem; padding: .55rem .9rem;
     border: 1px solid var(--pst-color-border, #e0e0e0); border-radius: 6px;
     cursor: pointer; user-select: none; transition: border-color .15s, background .15s;
   }
   .dcc:hover, .dcc.wsel { border-color: var(--pst-color-primary, #00897b); background: rgba(0,137,123,.06); }
   .dcc input { margin-top: 2px; accent-color: var(--pst-color-primary, #00897b);
                width: 16px; height: 16px; flex-shrink: 0; cursor: pointer; }
   .dcc strong { display: block; font-size: .9rem; font-weight: 600; }
   .dcc span   { font-size: .8rem; color: var(--pst-color-text-muted, #757575); }

   .dcsub { display: none; margin: .9rem 0 0 1rem; padding: .9rem 1.1rem;
            background: rgba(0,137,123,.04); border: 1px solid rgba(0,137,123,.2); border-radius: 6px; }
   .dcsub.won { display: block; }
   .dcsub .dcqt { font-size: .9rem; }
   .dcsub .dcqh { margin-bottom: .7rem; }
   .dcsub .dcc  { padding: .45rem .8rem; }

   .dcnav { display: flex; justify-content: space-between; align-items: center;
            margin-top: 1.4rem; padding-top: 1rem; border-top: 1px solid var(--pst-color-border, #e0e0e0); }
   .dcb { padding: .45rem 1.1rem; border-radius: 6px; border: none; font-size: .86rem;
          font-weight: 600; cursor: pointer; font-family: inherit; }
   .dcb-p { background: var(--pst-color-primary, #00897b); color: #fff; }
   .dcb-p:hover { filter: brightness(.85); }
   .dcb-s { background: transparent; color: var(--pst-color-text-muted, #757575);
            border: 1px solid var(--pst-color-border, #e0e0e0); }

   .dcerr { color: #c62828; font-size: .82rem; margin-top: .6rem; display: none; }
   .dcerr.won { display: block; }

   .dcres { border-radius: 6px; padding: 1rem 1.2rem; }
   .dcres.ok   { background: #e8f5e9; border-left: 4px solid #2e7d32; }
   .dcres.next { background: #e3f2fd; border-left: 4px solid #1565c0; }
   .dcres h4 { margin: 0 0 .5rem; font-size: .98rem; }
   .dcres.ok h4   { color: #2e7d32; }
   .dcres.next h4 { color: #1565c0; }
   .dcres p { font-size: .88rem; margin: .5rem 0 0; line-height: 1.6; }
   .dcres a { font-weight: 600; }
   .dcres ul { margin: .6rem 0 0 1.1rem; padding: 0; }
   .dcres li { font-size: .86rem; line-height: 1.55; margin-bottom: .45rem; }
   .dctag { display: inline-block; padding: .13rem .5rem; border-radius: 20px;
            font-size: .72rem; font-weight: 700; margin: 0 .28rem .3rem 0;
            background: #c8e6c9; color: #2e7d32; }
   </style>

   <button class="dcbtn" onclick="dcOpen()">&#128269; Is my data compatible?</button>

   <div class="dcov" id="dcov" onclick="if(event.target===this)dcClose()">
    <div class="dcmod">

     <div class="dchd">
       <h3>Data compatibility helper</h3>
       <button class="dcx" onclick="dcClose()" aria-label="Close">&times;</button>
     </div>

     <div class="dcpg won" id="dcp0">
       <div class="dcqt">Are you working with NWB files?</div>
       <div class="dcqh"><code>.nwb</code> files are self-contained &mdash; EthoGraph loads them directly.</div>
       <div class="dcch">
         <label class="dcc"><input type="radio" name="dcnwb" value="yes" onchange="dcRad(this,'nwb')">
           <div><strong>Yes &mdash; I have .nwb files</strong><span>From DANDI, NeuroConv, or another NWB pipeline</span></div></label>
         <label class="dcc"><input type="radio" name="dcnwb" value="no" onchange="dcRad(this,'nwb')">
           <div><strong>No &mdash; I have raw files</strong><span>Video, audio, pose, ephys, numpy&hellip;</span></div></label>
       </div>
       <div class="dcerr" id="dce0">Please select an option.</div>
     </div>

     <div class="dcpg" id="dcp1">
       <div class="dcqt">What do you want to visualise?</div>
       <div class="dcqh">Select everything that applies.</div>
       <div class="dcch">
         <label class="dcc"><input type="checkbox" onchange="dcChk(this,'video')">
           <div><strong>Videos</strong><span>.mp4 camera recordings</span></div></label>
         <label class="dcc"><input type="checkbox" onchange="dcChk(this,'pose')">
           <div><strong>Pose estimation</strong><span>DeepLabCut, SLEAP, LightningPose</span></div></label>
         <label class="dcc"><input type="checkbox" onchange="dcChk(this,'audio')">
           <div><strong>Audio / spectrogram</strong><span>.wav, .mp3, or video with sound</span></div></label>
         <label class="dcc"><input type="checkbox" onchange="dcChk(this,'ephys')">
           <div><strong>Electrophysiology</strong><span>Raw ephys or spike-sorted units (Kilosort)</span></div></label>
         <label class="dcc"><input type="checkbox" onchange="dcChk(this,'numpy')">
           <div><strong>Custom feature array</strong><span>Pre-computed signals as .npy</span></div></label>
         <label class="dcc"><input type="checkbox" onchange="dcChk(this,'other')">
           <div><strong>Other / custom format</strong><span>Not listed above</span></div></label>
       </div>

       <div class="dcsub" id="dcsubCam">
         <div class="dcqt">Multiple cameras?</div>
         <div class="dcqh">i.e. several video or pose files recorded simultaneously.</div>
         <div class="dcch">
           <label class="dcc"><input type="radio" name="dccam" value="single" onchange="dcRad(this,'cameras')">
             <div><strong>No &mdash; single camera</strong></div></label>
           <label class="dcc"><input type="radio" name="dccam" value="multi" onchange="dcRad(this,'cameras')">
             <div><strong>Yes &mdash; multiple cameras</strong></div></label>
         </div>
       </div>

       <div class="dcsub" id="dcsubAud">
         <div class="dcqt">How is your audio stored?</div>
         <div class="dcqh">All three work with drag &amp; drop; separate files per mic come with a caveat.</div>
         <div class="dcch">
           <label class="dcc"><input type="radio" name="dcaud" value="single" onchange="dcRad(this,'audio_setup')">
             <div><strong>Single microphone</strong></div></label>
           <label class="dcc"><input type="radio" name="dcaud" value="multichannel" onchange="dcRad(this,'audio_setup')">
             <div><strong>One multichannel file</strong><span>All mics in a single .wav</span></div></label>
           <label class="dcc"><input type="radio" name="dcaud" value="multi_files" onchange="dcRad(this,'audio_setup')">
             <div><strong>One file per microphone</strong></div></label>
         </div>
       </div>

       <div class="dcsub" id="dcsubLbl">
         <div class="dcqt">Have you already labelled this audio elsewhere?</div>
         <div class="dcqh">Existing annotations can be imported rather than redone.</div>
         <div class="dcch">
           <label class="dcc"><input type="radio" name="dclbl" value="no" onchange="dcRad(this,'labelled')">
             <div><strong>No &mdash; I'll label in EthoGraph</strong></div></label>
           <label class="dcc"><input type="radio" name="dclbl" value="yes" onchange="dcRad(this,'labelled')">
             <div><strong>Yes &mdash; in Audacity, Praat, evsonganaly&hellip;</strong>
             <span>Or another annotation tool</span></div></label>
         </div>
       </div>

       <div class="dcerr" id="dce1">Please answer every question above.</div>
     </div>

     <div class="dcpg" id="dcp2">
       <div class="dcqt">Does your data have a trial structure?</div>
       <div class="dcqh">Yes if you have separate files per trial, or one continuous recording in which you
       want to define trial windows.</div>
       <div class="dcch">
         <label class="dcc"><input type="radio" name="dctr" value="no" onchange="dcRad(this,'trials')">
           <div><strong>No &mdash; one continuous session</strong></div></label>
         <label class="dcc"><input type="radio" name="dctr" value="yes" onchange="dcRad(this,'trials')">
           <div><strong>Yes &mdash; trials or epochs</strong></div></label>
       </div>
       <div class="dcerr" id="dce2">Please select an option.</div>
     </div>

     <div class="dcpg" id="dcpr"><div id="dcresult"></div></div>

     <div class="dcnav">
       <button class="dcb dcb-s" id="dcback" onclick="dcPrev()" style="visibility:hidden">&larr; Back</button>
       <button class="dcb dcb-p" id="dcnext" onclick="dcNext()">Next &rarr;</button>
     </div>

    </div>
   </div>

   <script>
   (function(){
   var PREP='getting_started/preparing_data.html';
   var st={nwb:null,d:new Set(),cameras:null,audio_setup:null,labelled:null,trials:null},cur=0;

   function clearErr(){document.querySelectorAll('.dcerr').forEach(function(e){e.classList.remove('won');});}
   function show(id){document.querySelectorAll('.dcpg').forEach(function(p){p.classList.remove('won');});
                     document.getElementById(id).classList.add('won');}

   window.dcRad=function(i,k){
     document.querySelectorAll('input[name="'+i.name+'"]').forEach(function(o){o.closest('.dcc').classList.remove('wsel');});
     i.closest('.dcc').classList.add('wsel'); st[k]=i.value; clearErr();
   };
   window.dcChk=function(i,v){
     i.closest('.dcc').classList.toggle('wsel',i.checked);
     if(i.checked)st.d.add(v);else st.d.delete(v);
     var cam=st.d.has('video')||st.d.has('pose'), aud=st.d.has('audio');
     document.getElementById('dcsubCam').classList.toggle('won',cam);
     document.getElementById('dcsubAud').classList.toggle('won',aud);
     document.getElementById('dcsubLbl').classList.toggle('won',aud);
     if(!cam)st.cameras=null;
     if(!aud){st.audio_setup=null;st.labelled=null;}
     clearErr();
   };

   window.dcOpen=function(){document.getElementById('dcov').classList.add('won');};
   window.dcClose=function(){document.getElementById('dcov').classList.remove('won');dcReset();};

   function dcReset(){
     st={nwb:null,d:new Set(),cameras:null,audio_setup:null,labelled:null,trials:null};cur=0;
     document.querySelectorAll('.dcov input').forEach(function(i){i.checked=false;});
     document.querySelectorAll('.dcc').forEach(function(e){e.classList.remove('wsel');});
     document.querySelectorAll('.dcsub').forEach(function(e){e.classList.remove('won');});
     clearErr();show('dcp0');
     document.getElementById('dcback').style.visibility='hidden';
     var n=document.getElementById('dcnext');n.style.display='';n.textContent='Next →';
   }

   window.dcNext=function(){
     clearErr();
     if(cur===0){
       if(!st.nwb){document.getElementById('dce0').classList.add('won');return;}
       if(st.nwb==='yes')return result();
       cur=1;show('dcp1');document.getElementById('dcback').style.visibility='visible';return;
     }
     if(cur===1){
       var ok=st.d.size>0
         && !((st.d.has('video')||st.d.has('pose'))&&!st.cameras)
         && !(st.d.has('audio')&&(!st.audio_setup||!st.labelled));
       if(!ok){document.getElementById('dce1').classList.add('won');return;}
       if(st.d.has('other'))return result();
       cur=2;show('dcp2');document.getElementById('dcnext').textContent='See my setup →';return;
     }
     if(cur===2){
       if(!st.trials){document.getElementById('dce2').classList.add('won');return;}
       result();
     }
   };

   window.dcPrev=function(){
     clearErr();
     if(cur==='r'){cur=st.nwb==='yes'?0:(st.d.has('other')?1:2);}
     else cur=cur-1;
     show('dcp'+cur);
     var n=document.getElementById('dcnext');
     n.style.display='';n.textContent=cur===2?'See my setup →':'Next →';
     document.getElementById('dcback').style.visibility=cur>0?'visible':'hidden';
   };

   function result(){
     cur='r';show('dcpr');
     document.getElementById('dcback').style.visibility='visible';
     document.getElementById('dcnext').style.display='none';
     document.getElementById('dcresult').innerHTML=build();
   }

   function tags(){
     var m=[['video','Video'],['pose','Pose'],['audio','Audio'],
            ['ephys','Ephys'],['numpy','.npy'],['other','Other']],h='';
     m.forEach(function(p){if(st.d.has(p[0]))h+='<span class="dctag">'+p[1]+'</span>';});
     return h?'<div>'+h+'</div>':'';
   }

   function importNote(){
     if(st.labelled!=='yes')return '';
     return '<div class="dcres next" style="margin-top:.9rem"><h4>Import your existing annotations</h4>'
       +'<p>Use <em>File &rarr; Import labels&hellip;</em> and pick your format &mdash; the list appears once '
       +'an audio folder is loaded. EthoGraph reads these through '
       +'<a href="https://crowsetta.readthedocs.io/" target="_blank" rel="noopener">crowsetta</a>:</p>'
       +'<ul>'
       +'<li>Audacity label track &mdash; <code>aud-seq</code> (<code>.txt</code>)</li>'
       +'<li>Praat TextGrid &mdash; <code>textgrid</code> (<code>.TextGrid</code>)</li>'
       +'<li>evsonganaly &mdash; <code>notmat</code> (<code>.not.mat</code>)</li>'
       +'<li>SongAnnotationGUI &mdash; <code>yarden</code> (<code>.mat</code>)</li>'
       +'<li>TIMIT corpus &mdash; <code>timit</code> (<code>.phn</code>, <code>.wrd</code>)</li>'
       +'<li>Any CSV with <code>onset_s</code>, <code>offset_s</code>, <code>label</code> columns &mdash; '
       +'<code>simple-seq</code> (<code>.csv</code>, <code>.txt</code>)</li>'
       +'<li>crowsetta generic &mdash; <code>generic-seq</code> (<code>.csv</code>)</li>'
       +'</ul>'
       +'<p>Raven selection tables (<code>raven</code>) and Audacity spectrogram selections '
       +'(<code>aud-bbox</code>) are bounding-box formats, which the importer does not read &mdash; '
       +'re-export them as <code>simple-seq</code> instead.</p></div>';
   }

   function build(){
     if(st.nwb==='yes'){
       return '<div class="dcres ok"><h4>Drag &amp; drop your file</h4>'
         +'<div><span class="dctag">.nwb</span></div>'
         +'<p>Drop the <code>.nwb</code> onto the start page and click <strong>Load</strong>. '
         +'Trials, media references and features are read straight from the file.</p></div>';
     }
     if(st.d.has('other')){
       return '<div class="dcres next"><h4>Convert to a supported format</h4>'+tags()
         +'<p>EthoGraph has no built-in loader for your format. Pick whichever route fits your data:</p>'
         +'<ul>'
         +'<li><strong>Save as <code>.npy</code></strong> &mdash; a plain array of shape '
         +'<code>(n_samples, n_variables)</code>. Drag it onto the start page; a popup asks the '
         +'sampling rate. If it renders slowly, use the <strong>Downsample</strong> option in the I/O widget.</li>'
         +'<li><strong>Save as <code>.wav</code></strong> &mdash; best for high-rate periodic signals '
         +'(LFP, EMG, pressure). Convert with <code>audioio.write_audio()</code> and load it as audio: '
         +'min/max downsampling keeps the waveform and spectrogram fast with no manual downsampling.</li>'
         +'<li><strong>Write an xarray script</strong> &mdash; needed for arrays with more than two '
         +'dimensions. Wrap your data in an <code>xr.Dataset</code>; see '
         +'<a href="'+PREP+'">Preparing your own data</a>.</li>'
         +'</ul></div>'+importNote();
     }
     if(st.trials!=='yes'){
       var follow = st.d.has('pose')
         ? 'A follow-up popup asks the <strong>source software</strong> only if the pose format is ambiguous.'
         : st.d.has('numpy')
         ? 'A follow-up popup asks the <strong>sampling rate</strong> of the array. If a high-rate array '
           +'renders slowly, use the <strong>Downsample</strong> option in the I/O widget.'
         : 'No follow-up questions are needed.';
       var multi = (st.cameras==='multi'||st.audio_setup==='multi_files')
         ? '<p><strong>Multiple files per stream are fine</strong> &mdash; each video and its pose file '
           +'become <code>cam-1</code>, <code>cam-2</code>, &hellip; and each sound file becomes '
           +'<code>mic-1</code>, <code>mic-2</code>, &hellip;, no scripting needed. They must already be '
           +'<strong>temporally aligned</strong>: drag &amp; drop assumes every file starts at the same '
           +'moment. If your recordings started at different times, trim them to a common start, or define '
           +'per-stream offsets in an <a href="'+PREP+'">alignment file</a>.</p>'
         : '';
       return '<div class="dcres ok"><h4>Drag &amp; drop your files</h4>'+tags()
         +'<p>Drop these onto the start page and click <strong>Load</strong>. EthoGraph sorts them '
         +'by type and builds the alignment for you &mdash; no scripting. '+follow+'</p>'+multi+'</div>'
         +importNote();
     }
     return '<div class="dcres next"><h4>You need a session file</h4>'+tags()
       +'<p>Your data has a trial structure, which drag &amp; drop cannot infer &mdash; it handles one '
       +'recording session at a time. Build a session file plus an alignment file with a short Python '
       +'script &mdash; see <a href="'+PREP+'">Preparing your own data</a>.</p></div>'+importNote();
   }

   dcReset();
   })();
   </script>

Installation
------------

EthoGraph is installed with `uv <https://docs.astral.sh/uv/>`_, a fast Python package manager:

.. tab-set::

   .. tab-item:: macOS / Linux

      .. code-block:: bash

         curl -LsSf https://astral.sh/uv/install.sh | sh

   .. tab-item:: Windows

      .. code-block:: bash

         winget install astral-sh.uv

      Works from both PowerShell and Command Prompt; ``winget`` is built into Windows 11.

Then install the GUI as a standalone tool — one command, no environment to create or activate:

.. code-block:: bash

   uv tool install --python 3.12 "ethograph[gui,audio]"

To open the GUI, run:

.. code-block:: bash

   ethograph check # Linux only : Check for missing libraries
   ethograph launch

.. note::
   For installing into a dedicated virtual environment, optional extras, and **troubleshooting**, see the
   :doc:`installation guide <getting_started/installation>`.

After launching, there are some :doc:`example datasets <examples/index>` you can explore the GUI. To
learn more about all the functionalities, I recommend the
:doc:`user manual <getting_started/user_manual>`.

.. _target-support:

Support
-------

.. image:: _static/media/opensource.png
   :alt: Open-source projects EthoGraph depends on
   :align: left
   :width: 60%

EthoGraph is built on top of a number of open-source projects:
`PyAV <https://pyav.org/docs/stable/>`_,
`audioio <https://github.com/bendalab/audioio>`_,
`Neo <https://neo.readthedocs.io>`_,
`crowsetta <https://github.com/vocalpy/crowsetta>`_,
`Neurodata Without Borders <https://www.nwb.org/>`_,
`xarray <https://docs.xarray.dev/>`_,
`pynapple <https://pynapple.org/index.html>`_,
`movement <https://movement.neuroinformatics.dev/>`_,
`phy <https://github.com/cortex-lab/phy>`_,
`PyQtGraph <https://www.pyqtgraph.org/>`_, and
`pygfx <https://pygfx.org/>`_ (via
`pynaviz <https://github.com/pynapple-org/pynaviz>`_).
