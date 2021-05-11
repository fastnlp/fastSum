use strict;
use Test;

# check the behavior of accessing a text node that contains
# an entity. The value should be the entity name when
# NoExpand => 1.

my $loaded;
BEGIN { $| = 1; plan tests => 3; }
END   { ok(0) unless $loaded; }
require XML::DOM;
$loaded = 1;
ok(1);

  # set up
  my $parser = getParser();

  my $xml_string = <<XML;
<?xml version="1.0"?>
<!DOCTYPE document [
   <!ENTITY myEntityWithValue "should see entity here, not this string">
]>
<document>
  <test1 name="noEntity">some regular text data</test1>
  <test1 name="hasEntity">&myEntityWithValue;</test1>
</document>
XML

## TEST ##

# parse
my $doc = $parser->parse($xml_string);
my $root = $doc->getDocumentElement();

my @testNodes = getElementsByTagName($root, 'test1');

my $i = 0;
my @expected = ('some regular text data','&myEntityWithValue;');

foreach my $testNode (@testNodes) {
  # print it out
  $testNode->normalize;
  foreach my $child ($testNode->getChildNodes) {
    ok($child->getData, $expected[$i++]);
#    print STDERR "Test1 ==> text child of
# NODE:",$testNode->getAttribute('name')," has value: ", $child->getData, "\n";
  }
}

exit 0;

sub getParser {
  my ($my_string) = @_;

   my %options = (
                    NoExpand => 1,
                    ParseParamEnt => 0,
                 );

  my $parser = new XML::DOM::Parser(%options);
}

# convience method to return a list rather than nodeList
sub getElementsByTagName {
  my ($node, $tag) = @_;
  my @list;
  my $nodes = $node->getElementsByTagName($tag);
  my $numOfNodes= $nodes->getLength();
  for (my $i=0; $i< $numOfNodes; $i++) {
     push @list, $nodes->item($i);
  }
  return @list;
}

