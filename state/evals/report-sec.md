| model | task | n | score | skipped | ttft p50 s | total p50 s | prompt t/s | decode t/s | tokens | cached | reasoning | ctx | build | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| openai:gpt-5.6-terra | extract-full | 120 | 0.964 | 10 | 1.16 | 1.19 | - | - | 11075937 | 45771 | 1354 | 1050000 | openai/api | - |
| gpt-oss-120b | extract-full | 120 | 0.903 | 17 | 6.05 | 6.51 | 251 | 30 | 8788053 | 7955053 | 0 | 131072 | 82d6bb284d1f/? | - |
| gpt-oss-20b | extract-full | 120 | 0.767 | 17 | 3.74 | - | 604 | 47 | 8789554 | - | - | 131072 | 82d6bb284d1f/? | - |

> warning: task extract-full: runs use different configs (02b99ef64184, 24904fad12ea, d3bcf817b091)
