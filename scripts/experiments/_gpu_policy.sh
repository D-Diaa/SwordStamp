# Reserve whole GPUs for vLLM workers; other model workers may share.
needs_whole_gpu() {
  case "${1##*/}" in
    _gen.sh|_quality_batch.sh)               return 0 ;;
    _pmark_gen.sh|_pmark_detect.sh)          return 0 ;;
    _samark_gen.sh)                          return 0 ;;
    _attack.sh)
      # These paraphrasers each start a vLLM engine and need the whole GPU.
      case " $* " in
        *" attack.paraphraser=adaptive "*) return 0 ;;
        *" attack.paraphraser=custom "*)   return 0 ;;
        *" attack.paraphraser=oracle "*)   return 0 ;;
      esac
      return 1
      ;;
  esac
  return 1
}
