use strict;
use Test;

my $loaded;
BEGIN { $| = 1; plan tests => 5; }
END   { ok(0) unless $loaded; }
require XML::DOM;
$loaded = 1;
ok(1);

my $str = qq[<?xml version="1.0"?><test_doc>This is a simple test for XML::DOM::Text.</test_doc>];

# test 1 -- check for correct parsing of input string
my $parser = new XML::DOM::Parser;
my $doc = eval { $parser->parse($str); };
ok((not $@) && defined $doc);
 
# test 2 -- check for working splitText function
#   eval it because in splitText was a bug which kills perl
my $text     = $doc->getDocumentElement()->getFirstChild();
my $new_node = $text->splitText(10);
ok($text->getNodeValue, 'This is a ');
ok($new_node->getNodeValue, 'simple test for XML::DOM::Text.');
ok($text->getNextSibling, $new_node);
