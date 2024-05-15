$cities = @("bordeaux", "lille", "lyon", "marseille", "montpellier", "nantes", "nice", "paris", "strasbourg", "toulouse")

$MA_Pattern = "^Mean accuracy: ((-?)[0-9]+\.[0-9]+)"
$SD_Pattern = "^Standard deviation: ((-?)[0-9]+\.[0-9]+)"
$NumberPattern = '.*?((-?)[0-9]+\.[0-9]+).*'

$scores = @{}

mkdir -Force tmp_ai *>$null
 
foreach ($string in $cities) {
    $filePath = "best-models/v1_gbr_" + $string + ".sav.txt"
    if (Test-Path -Path $filePath) {
        $MA_Line = Get-Content -Path $filePath | Select-String -Pattern $MA_Pattern
        $MA = $MA_Line -replace $NumberPattern , '$1'
        $SD_Line = Get-Content -Path $filePath | Select-String -Pattern $SD_Pattern
        $SD = $SD_Line -replace $NumberPattern , '$1'
        $scores[$string] = @{"MA"=[double]$MA; "SD"=[double]$SD}
    } else {
        $scores[$string] = @{"MA"=0; "SD"=1}
    }
}

Write-Output($scores["MA"])

while ($scores.Values | Where-Object { $_["MA"] -lt 0.7 }) {
    foreach ($string in $cities) {
        if ([double]$scores[$string]["MA"] -ge 0.7) {
            continue
        } 
        Write-Output("Entrainement de " + $string)
        python3 .\main.py $string -t gbr *>$null
        
        $filePath = "tmp_ai/v1_gbr_" + $string + ".sav.txt"
        
        $MA_Line = Get-Content -Path $filePath | Select-String -Pattern $MA_Pattern
        $MA = $MA_Line -replace $NumberPattern , '$1'
        $SD_Line = Get-Content -Path $filePath | Select-String -Pattern $SD_Pattern
        $SD = $SD_Line -replace $NumberPattern , '$1'

        if ([double]$scores[$string]["MA"] -lt [double]$MA -and [double]$scores[$string]["SD"] -ge [double]$SD) {
            $scores[$string] = @{"MA"=[double]$MA; "SD"=[double]$SD}
            Write-Output("Nouveau meilleur score pour " + $string + " : " + $MA)
            Move-Item -Force ("tmp_ai/v1_gbr_" + $string + "*") ("best-models/")
        }

        Remove-Item ("tmp_ai/v1_gbr_" + $string + "*")
    }
}