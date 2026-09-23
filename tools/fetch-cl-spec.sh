#!/bin/sh
# fetch-cl-spec.sh --- Put cl-spec, at the revision this repository was checked against,
# where ASDF finds it.
#
# cl-spec is not in Quicklisp. Only the optional cl-gbdt/specs bundle and the
# cl-gbdt/tests/specs suite need it -- nothing a user of cl-gbdt loads does -- so this is
# a development/CI step, like ./tools/fetch-libs.sh. The revision is pinned rather than
# following cl-spec's main: a change there must not turn this repository's CI red without a
# commit here that chose it. Override with CL_SPEC_REF to try another revision.
#
# An existing checkout at the destination is left alone: on a development machine that is
# the developer's own clone, and this script does not own it. Its HEAD is printed either
# way; a HEAD other than the pin only warns, since a developer's own clone at another
# revision is theirs to keep. A destination that exists but is not a git checkout at all
# (git rev-parse HEAD fails) is an error, not silently accepted as "present".
set -eu

CL_SPEC_REF="${CL_SPEC_REF:-08d3adaf912b614537ca32cf5a9451fe0b807337}"
dest="${CL_SPEC_DIR:-$HOME/.roswell/local-projects/masatoi/cl-spec}"

if [ -e "$dest" ]; then
  head="$(git -C "$dest" rev-parse HEAD 2>/dev/null)" || {
    echo "fetch-cl-spec.sh: $dest exists but is not a git checkout (git -C \"$dest\" rev-parse HEAD failed)" >&2
    exit 1
  }
  if [ "$head" = "$CL_SPEC_REF" ]; then
    echo "cl-spec already present at $dest (HEAD $head); not touching it"
  else
    echo "fetch-cl-spec.sh: WARNING: cl-spec at $dest is at $head, not the pinned $CL_SPEC_REF; not touching it (this is your own clone)" >&2
  fi
  exit 0
fi

mkdir -p "$(dirname "$dest")"
git clone --quiet https://github.com/masatoi/cl-spec "$dest"
git -C "$dest" checkout --quiet "$CL_SPEC_REF"
echo "cl-spec $(git -C "$dest" rev-parse HEAD) at $dest"
