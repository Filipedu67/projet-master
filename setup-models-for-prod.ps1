$files = (Get-ChildItem .\best-models -Filter *.sav).Name

foreach ($file in $files) {
    $city = (($file -split '_')[2] -split '\.')[0]
    Copy-Item (".\best-models\" + $file) (".\ai-models\" + $city + '.sav')
}
