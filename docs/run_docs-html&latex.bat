call C:\ProgramData\Anaconda3\Scripts\activate.bat C:\ProgramData\Anaconda3
call make.bat latex
call build\latex\make.bat
copy build\latex\octseg.pdf build\OCTseg-docs.pdf /Y
call make.bat html
PAUSE